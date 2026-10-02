// Mirrors the serde output of shuvoice_core::settings and the settings bridge.

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }

export type Section = 'speech' | 'vocabulary' | 'typing' | 'text_to_speech' | 'audio' | 'appearance' | 'advanced'

export interface Choice {
  value: string
  label: string
}

export type FieldKind =
  | { type: 'bool' }
  | { type: 'int'; min: number; max: number }
  | { type: 'float'; min: number; max: number; step: number }
  | { type: 'text'; max_len: number }
  | { type: 'optional_text'; max_len: number }
  | { type: 'string_list'; item_max_len: number; max_items: number }
  | { type: 'string_map'; key_max_len: number; value_max_len: number; max_entries: number }
  | { type: 'choice'; choices: Choice[] }
  | { type: 'audio_device'; direction?: 'input' | 'output' }

export interface FieldMeta {
  id: string
  section: Section
  label: string
  help: string
  unit: string
  kind: FieldKind
  advanced?: boolean
  default?: Json
}

export interface Schema {
  sections: Section[]
  fields: FieldMeta[]
}

export interface SecretPresence {
  env: string
  present: boolean
  used_by: string
  source?: 'env' | 'local.dev' | 'local.env' | null
}

export interface Snapshot {
  path: string
  revision: string
  values: Record<string, Json>
  explicit: string[]
  config_error: string | null
  extra_choices: Record<string, Choice[]>
  secrets: SecretPresence[]
}

export interface FieldError {
  field: string | null
  message: string
}

export interface Applied {
  revision: string
  backup: string | null
  changed: string[]
}

export interface ServiceStatus {
  service: string
  active_state: string
  ui_ready: boolean | null
  stt: string | null
  tts: string | null
}

export interface InputDevice {
  index: number
  name: string
  channels: number
  default_sample_rate: number
}

export const SECTION_LABELS: Record<Section, string> = {
  speech: 'Speech',
  vocabulary: 'Vocabulary',
  advanced: 'Advanced',
  typing: 'Typing',
  text_to_speech: 'Text-to-Speech',
  audio: 'Audio',
  appearance: 'Appearance',
}

export type RestartOutcome =
  | { outcome: 'ready'; action: 'start' | 'restart' }
  | { outcome: 'starting'; action: string; message: string }
  | { outcome: 'handoff_failed' | 'action_failed' | 'readiness_failed'; action: string; message: string }
  | { outcome: 'unavailable' }
  | { outcome: 'not_active'; state: string }

export interface ApplyResult {
  saved: Applied
  restart: RestartOutcome
}
