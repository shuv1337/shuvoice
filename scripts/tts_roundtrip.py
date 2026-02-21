#!/usr/bin/env python3
"""Generate TTS fixtures and run round-trip ASR checks.

This script helps diagnose long-phrase reliability issues by:
1) synthesizing known text to WAV using espeak-ng/espeak
2) feeding the WAV through ShuVoice streaming ASR
3) printing reference vs hypothesis with similarity metrics

Usage examples:
  python scripts/tts_roundtrip.py --asr-backend nemo --device cuda
  python scripts/tts_roundtrip.py --asr-backend moonshine --moonshine-model-name moonshine/base
  python scripts/tts_roundtrip.py --phrases-file examples/tts_roundtrip_phrases.txt
  python scripts/tts_roundtrip.py --phrase "a custom sentence" --phrase "another"
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shuvoice.asr import create_backend
from shuvoice.asr_base import ASRBackend
from shuvoice.config import Config
from shuvoice.transcript import prefer_transcript

DEFAULT_PHRASES = [
    "This is a short sentence for baseline accuracy.",
    "This is so close to being a good application, but it still feels unreliable with longer phrases.",
    "We still have issues with recording cutting out on long sentences, and we need deterministic regression tests.",
    "Used for testing and returning back to speech to text, this sentence intentionally includes repeated structure.",
    "If this script catches truncation reliably, we can compare results before and after each change.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShuVoice TTS -> STT round-trip harness")
    parser.add_argument(
        "--output-dir",
        default="build/tts-roundtrip",
        help="Directory where generated WAV files and CSV report are written",
    )
    parser.add_argument(
        "--phrases-file",
        default=None,
        help="Optional newline-delimited phrase file (# comments supported)",
    )
    parser.add_argument(
        "--phrase",
        action="append",
        default=None,
        help="Provide one or more custom phrases directly",
    )
    parser.add_argument("--voice", default="en", help="espeak voice (default: en)")
    parser.add_argument(
        "--speed",
        type=int,
        default=170,
        help="espeak words per minute (default: 170)",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["nemo", "sherpa", "moonshine"],
        default=None,
        help="ASR backend override",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="ASR device override (NeMo backend)",
    )
    parser.add_argument(
        "--right-context",
        type=int,
        choices=[0, 1, 6, 13],
        default=None,
        help="NeMo right_context override",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="ASR model override",
    )
    parser.add_argument(
        "--sherpa-model-dir",
        default=None,
        help="Sherpa model directory containing tokens.txt + transducer ONNX artifacts",
    )
    parser.add_argument(
        "--sherpa-provider",
        choices=["cpu", "cuda"],
        default=None,
        help="Sherpa execution provider override",
    )
    parser.add_argument(
        "--sherpa-num-threads",
        type=int,
        default=None,
        help="Sherpa decoding threads override",
    )
    parser.add_argument(
        "--sherpa-chunk-ms",
        type=int,
        default=None,
        help="Sherpa chunk duration override (ms)",
    )
    parser.add_argument(
        "--moonshine-model-name",
        default=None,
        help="Moonshine model name override (e.g. moonshine/base)",
    )
    parser.add_argument(
        "--moonshine-model-dir",
        default=None,
        help="Local Moonshine model directory override",
    )
    parser.add_argument(
        "--moonshine-model-precision",
        default=None,
        help="Moonshine model precision override",
    )
    parser.add_argument(
        "--moonshine-chunk-ms",
        type=int,
        default=None,
        help="Moonshine chunk duration override (ms)",
    )
    parser.add_argument(
        "--moonshine-max-window-sec",
        type=float,
        default=None,
        help="Moonshine cumulative decode window override (seconds)",
    )
    parser.add_argument(
        "--moonshine-max-tokens",
        type=int,
        default=None,
        help="Moonshine max decode tokens override",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path (default: <output-dir>/roundtrip.csv)",
    )
    parser.add_argument(
        "--flush-chunks",
        type=int,
        default=3,
        help="Silent tail chunks to flush decoder (default: 3)",
    )
    return parser.parse_args()


def load_phrases(path: Path) -> list[str]:
    phrases: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        phrases.append(line)
    return phrases


def pick_tts_binary() -> str:
    for candidate in ("espeak-ng", "espeak"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "Neither 'espeak-ng' nor 'espeak' is available in PATH. "
        "Install with: sudo pacman -S espeak-ng"
    )


def synthesize_wav(tts_bin: str, text: str, output_path: Path, voice: str, speed: int):
    cmd = [tts_bin, "-v", voice, "-s", str(speed), "-w", str(output_path), text]
    subprocess.run(cmd, check=True)


def read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32), sample_rate


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    if audio.size == 0:
        return audio

    src_len = len(audio)
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))

    src_idx = np.arange(src_len, dtype=np.float64)
    dst_idx = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, audio).astype(np.float32)
    return out


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stream_transcribe(
    engine: ASRBackend,
    audio: np.ndarray,
    native_chunk: int,
    flush_chunks: int,
) -> str:
    engine.reset()

    transcript = ""
    offset = 0
    while offset < len(audio):
        chunk = audio[offset : offset + native_chunk]
        if len(chunk) < native_chunk:
            padded = np.zeros(native_chunk, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded
        text = engine.process_chunk(chunk.astype(np.float32))
        transcript = prefer_transcript(transcript, text)
        offset += native_chunk

    silence = np.zeros(native_chunk, dtype=np.float32)
    stable_steps = 0
    for _ in range(max(0, flush_chunks)):
        text = engine.process_chunk(silence)
        merged = prefer_transcript(transcript, text)
        if merged == transcript:
            stable_steps += 1
            if stable_steps >= 2:
                break
        else:
            stable_steps = 0
            transcript = merged

    return transcript.strip()


def main():
    args = parse_args()

    cfg = Config.load()
    if args.asr_backend:
        cfg.asr_backend = args.asr_backend
    if args.device:
        cfg.device = args.device
    if args.right_context is not None:
        cfg.right_context = args.right_context
    if args.model_name:
        cfg.model_name = args.model_name
    if args.sherpa_model_dir is not None:
        cfg.sherpa_model_dir = args.sherpa_model_dir
    if args.sherpa_provider:
        cfg.sherpa_provider = args.sherpa_provider
    if args.sherpa_num_threads is not None:
        cfg.sherpa_num_threads = args.sherpa_num_threads
    if args.sherpa_chunk_ms is not None:
        cfg.sherpa_chunk_ms = args.sherpa_chunk_ms
    if args.moonshine_model_name is not None:
        cfg.moonshine_model_name = args.moonshine_model_name
    if args.moonshine_model_dir is not None:
        cfg.moonshine_model_dir = args.moonshine_model_dir
    if args.moonshine_model_precision is not None:
        cfg.moonshine_model_precision = args.moonshine_model_precision
    if args.moonshine_chunk_ms is not None:
        cfg.moonshine_chunk_ms = args.moonshine_chunk_ms
    if args.moonshine_max_window_sec is not None:
        cfg.moonshine_max_window_sec = args.moonshine_max_window_sec
    if args.moonshine_max_tokens is not None:
        cfg.moonshine_max_tokens = args.moonshine_max_tokens

    cfg.__post_init__()

    phrases: list[str]
    if args.phrase:
        phrases = [p.strip() for p in args.phrase if p.strip()]
    elif args.phrases_file:
        phrases = load_phrases(Path(args.phrases_file))
    else:
        phrases = list(DEFAULT_PHRASES)

    if not phrases:
        raise RuntimeError("No phrases provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else output_dir / "roundtrip.csv"

    tts_bin = pick_tts_binary()

    print(f"Using TTS binary: {tts_bin}")
    print(f"ASR backend: {cfg.asr_backend}")
    if cfg.asr_backend == "nemo":
        print(f"ASR model: {cfg.model_name}")
        print(f"ASR device: {cfg.device}")
        print(f"Right context: {cfg.right_context}")
    elif cfg.asr_backend == "sherpa":
        print(f"Sherpa model dir: {cfg.sherpa_model_dir}")
        print(f"Sherpa provider: {cfg.sherpa_provider}")
        print(f"Sherpa chunk ms: {cfg.sherpa_chunk_ms}")
    else:
        print(f"Moonshine model name: {cfg.moonshine_model_name}")
        print(f"Moonshine model dir: {cfg.moonshine_model_dir}")
        print(f"Moonshine precision: {cfg.moonshine_model_precision}")
        print(f"Moonshine chunk ms: {cfg.moonshine_chunk_ms}")
        print(f"Moonshine max window sec: {cfg.moonshine_max_window_sec}")
        print(f"Moonshine max tokens: {cfg.moonshine_max_tokens}")
    print(f"Phrases: {len(phrases)}")
    print()

    engine = create_backend(cfg.asr_backend, cfg)
    engine.load()
    print(f"Native chunk samples: {engine.native_chunk_samples}")

    rows: list[dict[str, str]] = []

    for index, phrase in enumerate(phrases, start=1):
        wav_path = output_dir / f"phrase-{index:02d}.wav"
        synthesize_wav(tts_bin, phrase, wav_path, voice=args.voice, speed=args.speed)

        audio, sample_rate = read_wav_mono_float32(wav_path)
        audio = resample_linear(audio, sample_rate, cfg.sample_rate)

        hypothesis = stream_transcribe(
            engine,
            audio,
            native_chunk=engine.native_chunk_samples,
            flush_chunks=args.flush_chunks,
        )

        ref_norm = normalize_text(phrase)
        hyp_norm = normalize_text(hypothesis)
        similarity = difflib.SequenceMatcher(None, ref_norm, hyp_norm).ratio()

        rows.append(
            {
                "index": str(index),
                "audio": str(wav_path),
                "reference": phrase,
                "hypothesis": hypothesis,
                "similarity": f"{similarity:.3f}",
            }
        )

        print(f"[{index:02d}] similarity={similarity:.3f}")
        print(f"  REF: {phrase}")
        print(f"  HYP: {hypothesis}")
        print()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "audio", "reference", "hypothesis", "similarity"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote report: {csv_path}")


if __name__ == "__main__":
    main()
