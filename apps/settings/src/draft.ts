// Pure draft helpers. Rust owns validation and defaults; these only shape UI state.
import type { FieldError, FieldKind, FieldMeta, Json } from './schema.ts'

export type Values = Record<string, Json>

/** Changed fields only (draft vs. the loaded snapshot). */
export function diff(base: Values, draft: Values): Values {
  const changes: Values = {}
  for (const [id, value] of Object.entries(draft)) {
    if (!equal(base[id], value)) changes[id] = value
  }
  return changes
}

/** Field id → first message; draft-wide errors under `''`. */
export function errorsByField(errors: FieldError[]): Record<string, string> {
  const map: Record<string, string> = {}
  for (const { field, message } of errors) {
    const key = field ?? ''
    if (!(key in map)) map[key] = message
  }
  return map
}

export type Parsed = { ok: true; value: number } | { ok: false; message: string }

export function equal(a: Json | undefined, b: Json | undefined): boolean {
  if (a === b) return true
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every(k => Object.hasOwn(b, k) && equal((a as Record<string, Json>)[k], (b as Record<string, Json>)[k]))
}

export function searchFields(fields: FieldMeta[], query: string): FieldMeta[] {
  const normalize = (s: string) => s.toLowerCase().replace(/[-_]/g, ' ')
  const words = normalize(query).trim().split(/\s+/).filter(Boolean)
  return words.length ? fields.filter(f => words.every(w => normalize(`${f.label} ${f.id} ${f.help} ${f.section}`).includes(w))) : []
}

export function supports(hello: { features?: string[] } | null, feature: string): boolean {
  return hello?.features?.includes(feature) === true
}

/** Parse numeric text for an int/float field (range is checked by Rust). */
export function parseNumber(kind: FieldKind, text: string): Parsed {
  const trimmed = text.trim()
  if (trimmed === '') return { ok: false, message: 'Enter a number' }
  if (kind.type === 'int') {
    if (!/^-?\d+$/.test(trimmed)) return { ok: false, message: 'Enter a whole number' }
    return { ok: true, value: Number.parseInt(trimmed, 10) }
  }
  const value = Number(trimmed)
  if (!Number.isFinite(value)) return { ok: false, message: 'Enter a number' }
  return { ok: true, value }
}

export function formatNumber(kind: FieldKind, value: Json): string {
  if (typeof value !== 'number') return ''
  if (kind.type === 'float') return String(Math.round(value * 100) / 100)
  return String(value)
}

/** Step a number field by ±1 step, clamped to its range. */
export function step(kind: FieldKind, value: Json, direction: 1 | -1): number | null {
  if (typeof value !== 'number') return null
  if (kind.type === 'int') return Math.min(kind.max, Math.max(kind.min, value + direction))
  if (kind.type === 'float') {
    const next = Math.round((value + direction * kind.step) * 100) / 100
    return Math.min(kind.max, Math.max(kind.min, next))
  }
  return null
}

/** Hide fields that do not apply to the current draft. */
export function visible(field: FieldMeta, draft: Values): boolean {
  switch (field.id) {
    case 'asr.sherpa_profile':
      return draft['asr.asr_backend'] === 'sherpa'
    case 'tts.tts_backend':
    case 'tts.tts_default_voice_id':
    case 'tts.tts_playback_speed':
      return draft['tts.tts_enabled'] === true
    case 'tts.tts_kokoro_base_url':
      return draft['tts.tts_enabled'] === true && draft['tts.tts_backend'] === 'kokoro'
    default:
      return true
  }
}
