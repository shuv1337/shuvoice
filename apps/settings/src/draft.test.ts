import { describe, expect, test } from 'bun:test'
import { diff, errorsByField, formatNumber, parseNumber, step, visible } from './draft.ts'
import type { FieldMeta } from './schema.ts'

const int = { type: 'int', min: 8, max: 96 } as const
const float = { type: 'float', min: 0.5, max: 2, step: 0.05 } as const
const meta = (id: string): FieldMeta => ({ id, section: 'speech', label: id, help: '', unit: '', kind: { type: 'bool' } })

describe('draft', () => {
  test('diff reports only changed fields, including null', () => {
    expect(diff({ a: 1, b: 'x', c: 'dev' }, { a: 1, b: 'y', c: null })).toEqual({ b: 'y', c: null })
    expect(diff({ a: 1 }, { a: 1 })).toEqual({})
  })

  test('errorsByField keeps the first message and maps draft-wide errors to ""', () => {
    expect(
      errorsByField([
        { field: 'x', message: 'first' },
        { field: 'x', message: 'second' },
        { field: null, message: 'whole draft' },
      ]),
    ).toEqual({ x: 'first', '': 'whole draft' })
  })

  test('parseNumber distinguishes int and float input', () => {
    expect(parseNumber(int, ' 24 ')).toEqual({ ok: true, value: 24 })
    expect(parseNumber(int, '24.5').ok).toBe(false)
    expect(parseNumber(float, '1.25')).toEqual({ ok: true, value: 1.25 })
    expect(parseNumber(float, 'fast').ok).toBe(false)
    expect(parseNumber(float, '').ok).toBe(false)
  })

  test('step clamps to the range and avoids float drift', () => {
    expect(step(int, 96, 1)).toBe(96)
    expect(step(int, 20, -1)).toBe(19)
    expect(step(float, 1.2, 1)).toBe(1.25)
    expect(step(float, 0.5, -1)).toBe(0.5)
    expect(formatNumber(float, 1.2500000001)).toBe('1.25')
  })

  test('visibility follows backend choices', () => {
    expect(visible(meta('asr.sherpa_profile'), { 'asr.asr_backend': 'sherpa' })).toBe(true)
    expect(visible(meta('asr.sherpa_profile'), { 'asr.asr_backend': 'openai_realtime' })).toBe(false)
    const kokoro = { 'tts.tts_enabled': true, 'tts.tts_backend': 'kokoro' }
    expect(visible(meta('tts.tts_kokoro_base_url'), kokoro)).toBe(true)
    expect(visible(meta('tts.tts_kokoro_base_url'), { ...kokoro, 'tts.tts_backend': 'openai' })).toBe(false)
    expect(visible(meta('tts.tts_default_voice_id'), { 'tts.tts_enabled': false })).toBe(false)
    expect(visible(meta('overlay.font_size'), {})).toBe(true)
  })
})
