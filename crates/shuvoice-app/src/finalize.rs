#![allow(
    clippy::too_many_arguments,
    clippy::single_match_else,
    clippy::collapsible_if
)]
//! Generation-tagged finalize jobs (run off the session actor task).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use shuvoice_asr::{AsrError, FallbackOutcome};
use shuvoice_core::{
    FinalizationMode, STOP_TAIL_GRACE, TailFlushDecision, UtteranceState, apply_utterance_gain,
    evaluate_tail_flush_step, flush_noise_escalation, is_silent_utterance, make_flush_noise,
    observe_recording_chunk, prefer_transcript, sanitize_final_injection_text,
};
use tokio::sync::mpsc;
use tracing::{debug, info};

use crate::asr_owner::{AsrOwnerHandle, gen_is_current};
use crate::types::UtteranceGen;

/// Why a lifecycle reset was requested (applied on the actor via JobResult).
#[derive(Debug, Clone)]
pub enum LifecyclePurpose {
    /// PTT start — begin recording only after reset succeeds.
    StartRecording,
    /// One-shot recovery when starting while circuit is open.
    DisabledRecovery,
    /// Post-error recovery reset (non-CUDA).
    RecoverAfterFailure { context: String },
    /// Circuit-breaker cooldown elapsed.
    CircuitRecovery,
}

#[derive(Debug)]
pub enum JobResult {
    Chunk {
        utt_gen: UtteranceGen,
        text: Result<String, AsrError>,
    },
    Finalize {
        utt_gen: UtteranceGen,
        outcome: FinalizeOutcome,
    },
    /// ASR reset finished (generation-tagged by AsrOwnerHandle::reset).
    Reset {
        purpose: LifecyclePurpose,
        result: Result<UtteranceGen, AsrError>,
    },
    /// CUDA CPU fallback finished.
    Fallback {
        result: Result<FallbackOutcome, AsrError>,
    },
    /// Final text injection attempt (latch commit only on Ok).
    InjectCommit {
        utt_gen: UtteranceGen,
        text: String,
        attempt: u32,
        result: Result<(), String>,
    },
    /// Streaming partial injection (best-effort).
    InjectPartial {
        utt_gen: UtteranceGen,
        text: String,
        result: Result<(), String>,
    },
    /// Injector reset finished.
    InjectReset { result: Result<(), String> },
    /// Selection/clipboard capture for TTS.
    SelectionCapture {
        kind: TtsCaptureKind,
        result: Result<String, String>,
    },
}

/// TTS capture source for selection jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsCaptureKind {
    Selection,
    Clipboard,
}

#[derive(Debug)]
pub enum FinalizeOutcome {
    Silent,
    Committed {
        text: String,
    },
    Ready {
        text: String,
    },
    Cancelled,
    Failed {
        err: String,
        count_breaker: bool,
        cuda_recovered: bool,
    },
}

fn absorb_late_chunk(state: &mut UtteranceState, chunk: &[f32], wants_raw: bool) {
    observe_recording_chunk(state, chunk, wants_raw, 1, 0.15, 10.0);
}

/// Spawn finalize work. Collects late audio until `grace` elapses or cancel.
pub fn spawn_finalize_job(
    asr: AsrOwnerHandle,
    utt_gen: UtteranceGen,
    mut state: UtteranceState,
    mut late_rx: mpsc::UnboundedReceiver<Vec<f32>>,
    grace: Duration,
    cancel: Arc<AtomicBool>,
    min_speech_samples: usize,
    _speech_rms_threshold: f32,
    result_tx: mpsc::UnboundedSender<JobResult>,
    finish_timeout: Option<Duration>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let wants_raw = asr.wants_raw_audio();
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            if cancel.load(Ordering::Acquire) {
                debug!("finalize cancelled during grace");
                let _ = result_tx.send(JobResult::Finalize {
                    utt_gen,
                    outcome: FinalizeOutcome::Cancelled,
                });
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                while let Ok(chunk) = late_rx.try_recv() {
                    absorb_late_chunk(&mut state, &chunk, wants_raw);
                }
                break;
            }
            match tokio::time::timeout_at(deadline, late_rx.recv()).await {
                Ok(Some(chunk)) => absorb_late_chunk(&mut state, &chunk, wants_raw),
                Ok(None) => break,
                Err(_) => {
                    // Deadline hit: absorb any chunks already queued (lossless grace).
                    while let Ok(chunk) = late_rx.try_recv() {
                        absorb_late_chunk(&mut state, &chunk, wants_raw);
                    }
                    break;
                }
            }
        }

        if cancel.load(Ordering::Acquire) || !gen_is_current(&asr, utt_gen) {
            let _ = result_tx.send(JobResult::Finalize {
                utt_gen,
                outcome: FinalizeOutcome::Cancelled,
            });
            return;
        }

        if is_silent_utterance(&state, min_speech_samples) {
            // Logged at info (not debug) on purpose: the service runs RUST_LOG=info,
            // and a silently discarded utterance is indistinguishable from "nothing
            // happened" without these numbers. peak_rms below threshold means the
            // gate is misconfigured for this microphone, not that the user was quiet.
            info!(
                utt_gen,
                peak_rms = state.peak_rms,
                threshold = state.utterance_rms_threshold,
                speech_samples = state.speech_samples,
                min_speech_samples,
                total_samples = state.total,
                "utterance discarded as silent; nothing sent to ASR"
            );
            let _ = result_tx.send(JobResult::Finalize {
                utt_gen,
                outcome: FinalizeOutcome::Silent,
            });
            return;
        }

        let mode = asr.finalization_mode();
        let native = asr.native_chunk_samples();
        let outcome = match mode {
            FinalizationMode::OfflineInstant => {
                finalize_offline(&asr, utt_gen, &mut state, wants_raw, &cancel).await
            }
            FinalizationMode::RemoteManualCommit => {
                finalize_remote(
                    &asr,
                    utt_gen,
                    &mut state,
                    native,
                    wants_raw,
                    finish_timeout,
                    &cancel,
                )
                .await
            }
            FinalizationMode::LocalStreaming => {
                finalize_streaming(&asr, utt_gen, &mut state, native, wants_raw, &cancel).await
            }
        };
        let _ = result_tx.send(JobResult::Finalize { utt_gen, outcome });
    })
}

async fn finalize_offline(
    asr: &AsrOwnerHandle,
    utt_gen: UtteranceGen,
    state: &mut UtteranceState,
    wants_raw: bool,
    cancel: &AtomicBool,
) -> FinalizeOutcome {
    if cancel.load(Ordering::Acquire) {
        return FinalizeOutcome::Cancelled;
    }
    let mut audio = state.concatenated();
    if !wants_raw && state.utterance_gain > 1.05 {
        audio = apply_utterance_gain(&audio, state.utterance_gain);
    }
    match asr.process_utterance(utt_gen, audio).await {
        Ok((g, text)) if g == utt_gen && gen_is_current(asr, g) => {
            state.last_text = prefer_transcript(&state.last_text, &text);
            ready_text(state)
        }
        Ok(_) => FinalizeOutcome::Cancelled,
        Err(err) => map_err(asr, err).await,
    }
}

async fn finalize_remote(
    asr: &AsrOwnerHandle,
    utt_gen: UtteranceGen,
    state: &mut UtteranceState,
    native: usize,
    wants_raw: bool,
    finish_timeout: Option<Duration>,
    cancel: &AtomicBool,
) -> FinalizeOutcome {
    let mut failed = false;
    while state.total >= native {
        if cancel.load(Ordering::Acquire) {
            return FinalizeOutcome::Cancelled;
        }
        if !pump_native(asr, utt_gen, state, native, wants_raw).await {
            failed = true;
            break;
        }
    }
    if !failed && state.total > 0 {
        let audio = state.concatenated();
        match asr.process_chunk(utt_gen, audio).await {
            Ok((g, text)) if g == utt_gen => {
                state.last_text = prefer_transcript(&state.last_text, &text);
            }
            Ok(_) => return FinalizeOutcome::Cancelled,
            Err(err) => return map_err(asr, err).await,
        }
    }
    if !failed {
        match asr.finish_utterance(utt_gen, finish_timeout).await {
            Ok((g, text)) if g == utt_gen && gen_is_current(asr, g) => {
                state.last_text = prefer_transcript(&state.last_text, &text);
            }
            Ok(_) => return FinalizeOutcome::Cancelled,
            Err(err) => return map_err(asr, err).await,
        }
    }
    if failed {
        return FinalizeOutcome::Failed {
            err: "remote finalize failed".into(),
            count_breaker: false,
            cuda_recovered: false,
        };
    }
    ready_text(state)
}

async fn finalize_streaming(
    asr: &AsrOwnerHandle,
    utt_gen: UtteranceGen,
    state: &mut UtteranceState,
    native: usize,
    wants_raw: bool,
    cancel: &AtomicBool,
) -> FinalizeOutcome {
    while state.total >= native {
        if cancel.load(Ordering::Acquire) {
            return FinalizeOutcome::Cancelled;
        }
        if !pump_native(asr, utt_gen, state, native, wants_raw).await {
            break;
        }
    }
    if state.total > 0 && !cancel.load(Ordering::Acquire) {
        let remainder = state.concatenated();
        let mut padded = vec![0.0f32; native];
        let n = remainder.len().min(native);
        padded[..n].copy_from_slice(&remainder[..n]);
        if !wants_raw && state.utterance_gain > 1.05 {
            let gained = apply_utterance_gain(&padded[..n], state.utterance_gain);
            padded[..n].copy_from_slice(&gained);
        }
        if let Ok((g, text)) = asr.process_chunk(utt_gen, padded).await {
            if g == utt_gen {
                state.last_text = prefer_transcript(&state.last_text, &text);
            }
        }
    }
    let _ = tail_flush(asr, utt_gen, state, native, wants_raw, cancel).await;
    if cancel.load(Ordering::Acquire) || !gen_is_current(asr, utt_gen) {
        return FinalizeOutcome::Cancelled;
    }
    ready_text(state)
}

async fn pump_native(
    asr: &AsrOwnerHandle,
    utt_gen: UtteranceGen,
    state: &mut UtteranceState,
    native: usize,
    wants_raw: bool,
) -> bool {
    let (to_process, _) = match state.consume_native_chunk(native) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if to_process.is_empty() {
        return false;
    }
    let audio = if wants_raw {
        to_process
    } else {
        apply_utterance_gain(&to_process, state.utterance_gain)
    };
    match asr.process_chunk(utt_gen, audio).await {
        Ok((g, text)) if g == utt_gen => {
            state.last_text = prefer_transcript(&state.last_text, &text);
            true
        }
        _ => false,
    }
}

async fn tail_flush(
    asr: &AsrOwnerHandle,
    utt_gen: UtteranceGen,
    state: &mut UtteranceState,
    native: usize,
    wants_raw: bool,
    cancel: &AtomicBool,
) -> bool {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A5);
    let mut stable = 0usize;
    let mut ever = !state.last_text.trim().is_empty();
    let mut stalled = 0usize;
    for step in 0.. {
        if cancel.load(Ordering::Acquire) {
            return false;
        }
        let esc = flush_noise_escalation(stalled);
        let mut noise = make_flush_noise(native, 0.0, esc, &mut rng);
        if !wants_raw {
            noise = apply_utterance_gain(&noise, state.utterance_gain);
        }
        let text = match asr.process_chunk(utt_gen, noise).await {
            Ok((g, t)) if g == utt_gen => t,
            _ => break,
        };
        let merged = prefer_transcript(&state.last_text, &text);
        let changed = merged != state.last_text;
        if changed {
            state.last_text = merged;
            stalled = 0;
        } else {
            stalled += 1;
        }
        let (decision, next_stable, next_ever) =
            evaluate_tail_flush_step(false, changed, ever, stable, step);
        stable = next_stable;
        ever = next_ever;
        match decision {
            TailFlushDecision::Continue => {}
            TailFlushDecision::StopStable | TailFlushDecision::AbortNewRecording => break,
        }
    }
    true
}

fn ready_text(state: &UtteranceState) -> FinalizeOutcome {
    let text = state.last_text.trim();
    if text.is_empty() {
        return FinalizeOutcome::Silent;
    }
    let text = sanitize_final_injection_text(text);
    if text.is_empty() {
        FinalizeOutcome::Silent
    } else {
        FinalizeOutcome::Ready { text }
    }
}

async fn map_err(asr: &AsrOwnerHandle, err: AsrError) -> FinalizeOutcome {
    let msg = err.to_string();
    if shuvoice_core::looks_like_cuda_oom_error(&msg) || matches!(err, AsrError::CudaOom(_)) {
        if let Ok(FallbackOutcome::Applied { .. }) = asr.try_fallback_to_cpu().await {
            return FinalizeOutcome::Failed {
                err: msg,
                count_breaker: false,
                cuda_recovered: true,
            };
        }
    }
    FinalizeOutcome::Failed {
        err: msg,
        count_breaker: err.counts_for_breaker(),
        cuda_recovered: false,
    }
}

pub fn default_grace() -> Duration {
    STOP_TAIL_GRACE
}

pub fn spawn_chunk_job(
    asr: AsrOwnerHandle,
    utt_gen: UtteranceGen,
    pcm: Vec<f32>,
    result_tx: mpsc::UnboundedSender<JobResult>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let text = asr.process_chunk(utt_gen, pcm).await.map(|(_, t)| t);
        let _ = result_tx.send(JobResult::Chunk { utt_gen, text });
    })
}

/// Spawn ASR reset off the actor task (never await reset on the actor).
pub fn spawn_reset_job(
    asr: AsrOwnerHandle,
    purpose: LifecyclePurpose,
    result_tx: mpsc::UnboundedSender<JobResult>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let result = asr.reset().await;
        let _ = result_tx.send(JobResult::Reset { purpose, result });
    })
}

/// Spawn CUDA fallback off the actor task.
pub fn spawn_fallback_job(
    asr: AsrOwnerHandle,
    result_tx: mpsc::UnboundedSender<JobResult>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let result = asr.try_fallback_to_cpu().await;
        let _ = result_tx.send(JobResult::Fallback { result });
    })
}
