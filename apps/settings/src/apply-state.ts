import type { ApplyResult, Section } from './schema.ts'
import type { BridgeEvent } from './bridge.ts'

export type Page = Section | 'service' | 'shortcuts'

export type Step = 'validating' | 'waiting_idle' | 'reserving' | 'saving' | 'restarting'

export type Phase =
  | { kind: 'loading' }
  | { kind: 'idle' }
  | { kind: 'applying'; step: Step; busy: string[]; unlocked?: boolean }
  | { kind: 'done'; text: string; tone: 'ok' | 'warn' }
  | { kind: 'conflict' }
  | { kind: 'error'; message: string }

const BUSY_LABELS: Record<string, string> = {
  recording: 'dictation',
  processing: 'transcription',
  tts_synthesizing: 'read-aloud',
  tts_playing: 'read-aloud',
  tts_paused: 'paused read-aloud',
}

export function busyText(busy: string[]): string {
  const names = [...new Set(busy.map((b) => BUSY_LABELS[b] ?? b))]
  return `Waiting for ${names.join(' and ') || 'ShuVoice'} to finish…`
}

export function progressPhase(event: BridgeEvent, previous: Phase): Phase {
  if (event.event !== 'progress' || !event.phase) return previous
  return {
    kind: 'applying',
    step: event.phase as Step,
    busy: event.busy ?? [],
    unlocked:
      event.reservation === 'unsupported' || (previous.kind === 'applying' && previous.unlocked),
  }
}

export function progressText(phase: Extract<Phase, { kind: 'applying' }>): string {
  if (phase.unlocked) return 'Applying without lock (older service)'
  switch (phase.step) {
    case 'waiting_idle':
      // The bridge announces this phase before its first idle check.
      return phase.busy.length > 0 ? busyText(phase.busy) : 'Checking ShuVoice is idle…'
    case 'reserving':
      return 'Reserving ShuVoice…'
    case 'restarting':
      return 'Restarting ShuVoice…'
    case 'validating':
      return 'Checking settings…'
    case 'saving':
      return 'Saving…'
  }
}

/** Footer message for a finished apply. Everything here has been saved. */
export function outcomePhase(result: ApplyResult): Phase {
  const r = result.restart
  switch (r.outcome) {
    case 'ready':
      return { kind: 'done', text: 'Applied ✓', tone: 'ok' }
    case 'starting':
      return { kind: 'done', text: 'Saved · ShuVoice is still starting', tone: 'warn' }
    case 'handoff_failed':
    case 'action_failed':
    case 'readiness_failed':
      return { kind: 'error', message: `Saved, but ${r.message}` }
    case 'unavailable':
    case 'not_active':
      return { kind: 'done', text: 'Saved · restart ShuVoice to apply', tone: 'warn' }
  }
}
