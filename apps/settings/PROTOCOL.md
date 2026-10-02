# Settings bridge protocol (v1)

`shuvoice settings-bridge` speaks JSON lines on stdio. One request per line;
one response per request, optionally preceded by progress events with the same
`id`. Stdout carries protocol only. Secret values never appear.

```text
-> {"v":1,"id":7,"op":"<op>","params":{…}}
<- {"v":1,"id":7,"event":"progress","phase":"…",…}        (0..n, long ops only)
<- {"v":1,"id":7,"ok":true,"result":{…}}
<- {"v":1,"id":7,"ok":false,"error":{"kind":"…","message":"…",…}}
```

Error kinds: `protocol`, `unknown_op`, `io`, `invalid` (`errors: FieldError[]`),
`conflict` (`current_revision`), `busy` (`busy?: string[]`), `cancelled`,
`unavailable`.

All additions below are additive within v1. `hello.result.features` lists the
optional ops the bridge supports so the app can feature-detect.

## Values and fields

Field ids are `section.key` of real config keys (one virtual id:
`asr.sherpa_profile`). Values are JSON: `null` means "unset → use default".

`schema` → `{ sections: Section[], fields: FieldMeta[], excluded: {id, reason}[] }`

```ts
type Section = 'speech' | 'vocabulary' | 'typing' | 'text_to_speech' | 'audio'
             | 'appearance' | 'advanced'
interface FieldMeta {
  id: string; section: Section; label: string; help: string; unit: string
  advanced: boolean          // shown under the section's collapsed "Advanced" group
  default: Json              // effective default, for "Reset to default"
  kind: FieldKind
}
type FieldKind =
  | { type: 'bool' }
  | { type: 'int'; min: number; max: number }
  | { type: 'float'; min: number; max: number; step: number }
  | { type: 'text'; max_len: number }
  | { type: 'optional_text'; max_len: number }            // null = unset
  | { type: 'choice'; choices: {value: string; label: string}[] }
  | { type: 'audio_device'; direction: 'input' | 'output' } // null | name | index
  | { type: 'string_list'; item_max_len: number; max_items: number }
  | { type: 'string_map'; key_max_len: number; value_max_len: number; max_entries: number }
```

Every key in `config_section_fields()` is either a field or listed in
`excluded` with a reason (a Rust test enforces this).

## Existing ops

`hello`, `schema`, `snapshot`, `validate {changes}`, `save {revision, changes}`,
`status`, `devices` (input; result also has `outputs: InputDevice[]` when the
bridge supports `output_devices`), `apply {revision, changes, onboarding?}`,
`cancel`.

`snapshot` result: `{path, revision, values, explicit, config_error,
extra_choices, secrets: {env, present, used_by, source: 'env'|'local.dev'|'local.env'|null}[]}`.

### `apply` (race-free)

Phases, in order: `validating` → `waiting_idle` (`busy: string[]`, repeats
when the busy set changes; cancellable) → `reserving` → `saving` →
`restarting` → result. The bridge takes a maintenance reservation from the
running service before saving, so no recording/read-aloud can start between
the idle check and the restart. A stopped service skips the reservation.

Result: `{ saved: Applied, restart: RestartOutcome }`.
`RestartOutcome.outcome`: `ready` | `starting` | `handoff_failed` |
`action_failed` | `readiness_failed` | `unavailable` | `not_active`.
Readiness is satisfied only by a new service invocation reporting `ui_ready`.
This confirms invocation/socket freshness, not the loaded config revision:
`debug_status` does not yet expose a startup config revision. An external editor
that ignores the writer lock can still change the config before startup loads it.

The service uses additive control commands `maintenance_reserve` (returns
`OK reserved token=<u64> ttl=120`) and `maintenance_release <token>`.
Only the owning token releases a reservation; expired/stale tokens are rejected.
The bridge cancels before saving and releases on disconnect/failure. After the
atomic save begins, cancellation cannot undo the commit; restart completes.
If saving consumes 60 seconds of the 120-second lease, it reports saved-but-not-
restarted instead of risking an expired reservation. Readiness checks systemd's
new InvocationID against the ID in `debug_status`, rejecting old sockets.

`params.onboarding: true`: after a successful save, also writes the
setup-complete marker, and starts (or restarts) the service.

## New ops

| op | params | result |
|---|---|---|
| `onboarding_defaults` | — | `{ values: Record<id, Json> }` — current snapshot values with the wizard defaults applied only to keys not explicitly set (fresh config → Parakeet CPU instant, Kokoro 1.25×). Saves nothing. |
| `capabilities` | `{ changes }` | `{ vocabulary_hints: { supported: boolean, detail: string } }` for the draft (saved config + changes). |
| `corrections_preview` | `{ text, changes }` | `{ output: string, builtins: Record<string,string> }` — runs Rust's real post-processing (replacements incl. built-ins, case, capitalization) for the draft. |
| `shortcut_get` | — | `{ current: {id, label} \| null, options: {id, label}[], config_path: string \| null, error: string \| null }` |
| `shortcut_set` | `{ id, dry_run }` | `{ status: 'added' \| 'already_present' \| 'replaced' \| 'unsupported' \| 'error', message, conflicts: string[], backup: string \| null }` — edits only ShuVoice's managed binding in the Hyprland config; `dry_run` reports without writing. |
| `models` | `{ changes }` | `{ required: { id, label, installed: boolean, size_hint: string \| null }[] }` — models the draft needs. |
| `model_download` | `{ id }` | progress events `{phase:'downloading', fraction: number \| null, text}`; result `{ id, installed: true }`. Shares the worker slot with `apply` (concurrent → `busy`); `cancel` stops it. |

Model IDs are opaque: call `models` for the current draft before `model_download`.
The bridge keeps that validated inventory and destination for the download.
Sherpa and curated Piper downloads reuse the setup downloaders. NeMo/Moonshine
models remain worker-managed and return `unavailable` for standalone download;
Piper requires its runtime already installed (this op never installs packages).
Lua Hyprland configs and external includes that cannot be inspected safely
return `unsupported` for shortcut edits; no Lua/config files are rewritten.
For simultaneous raw Sherpa keys and `asr.sherpa_profile`, the preset is applied
first and explicit raw key changes win. Null removes a key and restores its
core default (including the virtual profile's mapped keys).

## App launch

`shuvoice-settings [--onboarding]`. `--onboarding` opens the onboarding flow
(engine + model download, microphone, read-aloud, shortcut, then
`apply {onboarding:true}`) prefilled from `onboarding_defaults`.
