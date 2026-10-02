import { expect, test } from 'bun:test'
import { progressPhase, progressText } from './apply-state.ts'

test('unsupported reservation continues applying with neutral status until completion', () => {
  const phase = progressPhase(
    { event: 'progress', phase: 'reserving', reservation: 'unsupported' },
    { kind: 'idle' },
  )
  expect(phase.kind).toBe('applying')
  if (phase.kind !== 'applying') throw new Error('Expected apply progress')
  expect(progressText(phase)).toBe('Applying without lock (older service)')
  const saving = progressPhase({ event: 'progress', phase: 'saving' }, phase)
  expect(saving).toMatchObject({ kind: 'applying', step: 'saving', unlocked: true })
  const next = progressPhase({ event: 'progress', phase: 'validating' }, { kind: 'idle' })
  expect(next).toMatchObject({ unlocked: false })
})

test('idle check reads as a check, busy reads as waiting', () => {
  const base = { kind: 'applying', step: 'waiting_idle', unlocked: false } as const
  expect(progressText({ ...base, busy: [] })).toBe('Checking ShuVoice is idle…')
  expect(progressText({ ...base, busy: ['recording'] })).toContain('Waiting for')
})
