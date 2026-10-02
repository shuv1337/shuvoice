// Contract-only window fixture. No service, filesystem or Hyprland writes.
import { createRenderer, enableAutomation, render } from '@gpuix/react'
import { App } from '../app.tsx'
import { Bridge } from '../bridge.ts'
import { loadBrand } from '../branding.ts'
import type { FieldMeta, Snapshot } from '../schema.ts'

const fields: FieldMeta[] = [
  { id: 'asr.asr_backend', section: 'speech', label: 'Engine', kind: { type: 'choice', choices: [{ value: 'sherpa', label: 'Parakeet CPU' }] }, default: 'sherpa' },
  { id: 'vocabulary.terms', section: 'vocabulary', label: 'Preferred terms', kind: { type: 'string_list', item_max_len: 100, max_items: 100 }, default: [] },
  { id: 'typing.text_replacements', section: 'vocabulary', label: 'Correction rules', kind: { type: 'string_map', key_max_len: 100, value_max_len: 100, max_entries: 100 }, default: {} },
  { id: 'audio.input_device', section: 'audio', label: 'Microphone', kind: { type: 'audio_device', direction: 'input' }, default: null },
  { id: 'audio.output_device', section: 'audio', label: 'Speaker', kind: { type: 'audio_device', direction: 'output' }, default: null },
  { id: 'tts.tts_enabled', section: 'text_to_speech', label: 'Read aloud', kind: { type: 'bool' }, default: true },
  { id: 'tts.tts_default_voice_id', section: 'text_to_speech', label: 'Voice', kind: { type: 'optional_text', max_len: 100 }, default: null },
  { id: 'overlay.font_size', section: 'appearance', label: 'Caption size', kind: { type: 'int', min: 10, max: 80 }, default: 20 },
  { id: 'typing.delay', section: 'typing', label: 'Delay', kind: { type: 'float', min: 0, max: 2, step: 0.1 }, default: 0.1 },
  { id: 'advanced.path', section: 'advanced', label: 'Model path', kind: { type: 'optional_text', max_len: 100 }, default: null, advanced: true },
].map(f => ({ help: '', unit: '', advanced: false, ...f })) as FieldMeta[]
let snapshot: Snapshot = { path: '/fixture/config.toml', revision: 'r1', explicit: [], config_error: null, extra_choices: {}, secrets: [{ env: 'OPENAI_API_KEY', present: true, used_by: 'asr', source: 'local.dev' }], values: Object.fromEntries(fields.map(f => [f.id, f.default ?? null])) }
snapshot.values['vocabulary.terms'] = ['ShuVoice', 'Hyprland']
snapshot.values['typing.text_replacements'] = { 'shoe voice': 'ShuVoice' }
let line = (_: string) => {}
let installed = false
const features = ['onboarding_defaults', 'capabilities', 'corrections_preview', 'shortcut_get', 'shortcut_set', 'models', 'model_download', 'output_devices']
const bridge = new Bridge({
  onLine(fn) { line = fn }, onClose() {}, close() {},
  send(raw) {
    const { id, op, params } = JSON.parse(raw)
    const reply = (result: unknown) => line(JSON.stringify({ v: 1, id, ok: true, result }))
    queueMicrotask(() => {
      switch (op) {
        case 'hello': reply({ features }); break
        case 'schema': reply({ sections: ['speech', 'vocabulary', 'audio', 'text_to_speech', 'typing', 'appearance', 'advanced'], fields }); break
        case 'snapshot': reply(snapshot); break
        case 'onboarding_defaults': reply({ values: snapshot.values }); break
        case 'status': reply({ active_state: 'inactive', ui_ready: false, stt: 'idle', tts: 'idle' }); break
        case 'devices': reply({ devices: [{ index: 0, name: 'Microphone' }], outputs: [{ index: 0, name: 'Speakers' }] }); break
        case 'capabilities': reply({ vocabulary_hints: { supported: false, detail: 'Parakeet does not use hints. Correction rules still apply.' } }); break
        case 'corrections_preview': reply({ output: params.text.replaceAll('shoe voice', 'ShuVoice'), builtins: { 'new line': '\n' } }); break
        case 'models': reply({ required: [{ id: 'parakeet', label: 'Parakeet', installed, size_hint: '600 MB' }] }); break
        case 'model_download': line(JSON.stringify({ id, event: 'progress', phase: 'downloading', fraction: 0.5, text: 'Downloading Parakeet' })); setTimeout(() => { installed = true; reply({ id: 'parakeet', installed: true }) }, 400); break
        case 'shortcut_get': reply({ current: { id: 'super-v', label: 'Super + V' }, options: [{ id: 'super-v', label: 'Super + V' }], config_path: '/fixture/hypr.conf', error: null }); break
        case 'shortcut_set': reply({ status: 'already_present', message: params.dry_run ? 'Binding already present' : 'Shortcut confirmed', conflicts: [], backup: params.dry_run ? null : '/fixture/hypr.conf.bak' }); break
        case 'apply': snapshot = { ...snapshot, values: { ...snapshot.values, ...params.changes }, revision: 'r2' }; line(JSON.stringify({ id, event: 'progress', phase: 'reserving' })); setTimeout(() => reply({ saved: { revision: 'r2', changed: Object.keys(params.changes), backup: '/fixture/config.bak' }, restart: { outcome: 'ready', action: 'start' } }), 100); break
        default: reply({})
      }
    })
  },
})
const renderer = createRenderer()
const options = { title: 'ShuVoice Settings · Fixture', appId: 'shuvoice-settings', width: 1000, height: 760 }
renderer.init(options)
enableAutomation(renderer)
render(<App bridge={bridge} brand={await loadBrand()} onboarding={process.argv.includes('--onboarding')} />, { ...options, renderer })
