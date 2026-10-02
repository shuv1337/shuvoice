import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { AnimatePresence, motion } from '@gpuix/react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@gpuix/react/select'
import { type Bridge, BridgeError, type BridgeEvent } from './bridge.ts'
import type { Brand } from './branding.ts'
import { diff, errorsByField, formatNumber, parseNumber, step, visible, searchFields, supports, type Values } from './draft.ts'
import {
  SECTION_LABELS,
  type ApplyResult,
  type Choice,
  type FieldError,
  type FieldMeta,
  type InputDevice,
  type Json,
  type Schema,
  type Section,
  type ServiceStatus,
  type Snapshot,
} from './schema.ts'

const C = {
  bg: '#16161a',
  side: '#1c1c21',
  card: '#24242b',
  cardHover: '#2b2b33',
  field: '#111114',
  line: '#2f2f38',
  lineStrong: '#44444f',
  text: '#e6e6ea',
  dim: '#9a9aa6',
  faint: '#9a9aa6',
  accent: '#7aa2f7',
  onAccent: '#101014',
  ok: '#9ece6a',
  warn: '#e0af68',
  bad: '#f7768e',
}

type Page = Section | 'service' | 'shortcuts'

type Step = 'validating' | 'waiting_idle' | 'reserving' | 'saving' | 'restarting'

type Phase =
  | { kind: 'loading' }
  | { kind: 'idle' }
  | { kind: 'applying'; step: Step; busy: string[] }
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

function busyText(busy: string[]): string {
  const names = [...new Set(busy.map((b) => BUSY_LABELS[b] ?? b))]
  return `Waiting for ${names.join(' and ') || 'ShuVoice'} to finish…`
}

/** Footer message for a finished apply. Everything here has been saved. */
function outcomePhase(result: ApplyResult): Phase {
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

const DEFAULT_DEVICE = '__default__'

function Text({ children, color = C.text, size, bold, testId }: {
  children: ReactNode
  color?: string
  size?: number
  bold?: boolean
  testId?: string
}) {
  return (
    <text testId={testId} style={{ color, fontSize: size, fontWeight: bold ? 'bold' : undefined }}>
      {children}
    </text>
  )
}

function Button({ label, onClick, primary, disabled, testId }: {
  label: string
  onClick: () => void
  primary?: boolean
  disabled?: boolean
  testId?: string
}) {
  const bg = primary ? C.accent : C.card
  return (
    <div
      testId={testId}
      tabIndex={disabled ? -1 : 0}
      onKeyDown={e => { if (!disabled && (e.key === 'enter' || e.key === 'space')) onClick() }}
      onClick={disabled ? undefined : onClick}
      style={{
        paddingLeft: 14,
        paddingRight: 14,
        paddingTop: 8,
        paddingBottom: 8,
        borderRadius: 6,
        cursor: disabled ? 'default' : 'pointer',
        opacity: disabled ? 0.45 : 1,
        backgroundColor: bg,
        hover: disabled ? undefined : { backgroundColor: primary ? '#8fb2fa' : C.cardHover },
      }}
    >
      <Text color={primary ? C.onAccent : C.text} bold={primary}>
        {label}
      </Text>
    </div>
  )
}

const CHEVRON = (color: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`
const CHECK = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="${C.accent}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`

const CONTROL_WIDTH = 320

function ChoiceControl({ field, value, choices, onValue }: {
  field: FieldMeta
  value: Json
  choices: Choice[]
  onValue: (value: string) => void
}) {
  return (
    <Select
      value={typeof value === 'string' ? value : undefined}
      onValueChange={onValue}
      items={choices.map((c) => ({ value: c.value, label: c.label, textValue: c.label }))}
    >
      <SelectTrigger
        testId={`field-${field.id}`}
        style={({ open }) => ({
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          gap: 8,
          width: CONTROL_WIDTH,
          paddingLeft: 10,
          paddingRight: 8,
          paddingTop: 8,
          paddingBottom: 8,
          borderRadius: 6,
          borderWidth: 1,
          borderColor: open ? C.accent : C.lineStrong,
          backgroundColor: C.card,
          cursor: 'pointer',
          hover: { backgroundColor: C.cardHover },
        })}
      >
        <div style={{ flexGrow: 1, minWidth: 0 }}>
          <SelectValue style={{ color: C.text }} placeholder={<Text color={C.dim}>Choose…</Text>} />
        </div>
        <svg source={CHEVRON(C.dim)} style={{ width: 16, height: 16, flexShrink: 0 }} />
      </SelectTrigger>
      <SelectContent
        sideOffset={4}
        align="start"
        style={{
          width: CONTROL_WIDTH,
          backgroundColor: C.card,
          borderRadius: 6,
          borderWidth: 1,
          borderColor: C.lineStrong,
          padding: 4,
          boxShadow: { offsetX: 0, offsetY: 8, blurRadius: 24, spreadRadius: 0, color: '#00000099' },
        }}
      >
        {choices.map((c) => (
          <SelectItem
            key={c.value}
            value={c.value}
            testId={`option-${field.id}-${c.value}`}
            style={({ highlighted }) => ({
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              gap: 8,
              paddingLeft: 8,
              paddingRight: 8,
              paddingTop: 6,
              paddingBottom: 6,
              borderRadius: 4,
              cursor: 'pointer',
              backgroundColor: highlighted ? C.line : C.card,
            })}
          >
            {({ selected, highlighted }) => (
              <>
                <div style={{ flexGrow: 1, minWidth: 0 }}>
                  <Text color={selected || highlighted ? C.text : C.dim}>{c.label}</Text>
                </div>
                <div style={{ width: 16, height: 16, flexShrink: 0 }}>
                  {selected ? <svg source={CHECK} style={{ width: 16, height: 16 }} /> : null}
                </div>
              </>
            )}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

const INPUT_STYLE = {
  padding: 8,
  minHeight: 36,
  flexShrink: 0,
  borderRadius: 6,
  borderWidth: 1,
  borderColor: C.line,
  backgroundColor: C.field,
  color: C.text,
} as const

function Switch({ field, value, onValue }: { field: FieldMeta; value: Json; onValue: (value: boolean) => void }) {
  const on = value === true
  return (
    <div
      testId={`field-${field.id}`}
      onClick={() => onValue(!on)}
      tabIndex={0}
      onKeyDown={e => { if (e.key === 'enter' || e.key === 'space') onValue(!on) }}
      style={{
        width: 44,
        height: 24,
        borderRadius: 12,
        padding: 3,
        cursor: 'pointer',
        backgroundColor: on ? C.accent : C.line,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: on ? 'flex-end' : 'flex-start',
      }}
    >
      <div style={{ width: 18, height: 18, borderRadius: 9, backgroundColor: on ? C.onAccent : C.dim }} />
    </div>
  )
}

function NumberControl({ field, value, onValue, onInvalid }: {
  field: FieldMeta
  value: Json
  onValue: (value: number) => void
  onInvalid: (message: string) => void
}) {
  const [text, setText] = useState(formatNumber(field.kind, value))
  useEffect(() => {
    const parsed = parseNumber(field.kind, text)
    if (!parsed.ok || parsed.value !== value) setText(formatNumber(field.kind, value))
    // Resync only when the value changes from outside this input.
  }, [value])

  const bump = (direction: 1 | -1) => {
    const next = step(field.kind, value, direction)
    if (next !== null) onValue(next)
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 6 }}>
      <Button label="−" onClick={() => bump(-1)} testId={`dec-${field.id}`} />
      <input
        testId={`field-${field.id}`}
        value={text}
        onChange={(e) => {
          const next = e.value ?? ''
          setText(next)
          const parsed = parseNumber(field.kind, next)
          if (parsed.ok) onValue(parsed.value)
          else onInvalid(parsed.message)
        }}
        style={{ ...INPUT_STYLE, width: 96 }}
      />
      <Button label="+" onClick={() => bump(1)} testId={`inc-${field.id}`} />
      {field.unit ? <Text color={C.dim}>{field.unit}</Text> : null}
    </div>
  )
}

function TextControl({ field, value, onValue }: { field: FieldMeta; value: Json; onValue: (value: string) => void }) {
  return (
    <input
      testId={`field-${field.id}`}
      value={typeof value === 'string' ? value : ''}
      onChange={(e) => onValue(e.value ?? '')}
      style={{ ...INPUT_STYLE, width: 420 }}
    />
  )
}

function deviceChoices(devices: InputDevice[] | null, value: Json): { selected: string; choices: Choice[] } {
  const choices: Choice[] = [{ value: DEFAULT_DEVICE, label: 'System default' }]
  for (const d of devices ?? []) choices.push({ value: `name:${d.name}`, label: d.name })
  let selected = DEFAULT_DEVICE
  if (typeof value === 'string') {
    selected = `name:${value}`
    if (!choices.some((c) => c.value === selected)) choices.push({ value: selected, label: `${value} (not found)` })
  } else if (typeof value === 'number') {
    const match = devices?.find((d) => d.index === value)
    selected = match ? `name:${match.name}` : `index:${value}`
    if (!match) choices.push({ value: selected, label: `Device #${value} (not found)` })
  }
  return { selected, choices }
}

function CollectionControl({ field, value, onValue, onInvalid }: { field: FieldMeta; value: Json; onValue: (v: Json) => void; onInvalid: (m: string) => void }) {
  const map = field.kind.type === 'string_map'
  const decode = (v: Json): string[][] => map ? Object.entries(v && !Array.isArray(v) && typeof v === 'object' ? v : {}).map(([k, v]) => [k, String(v)]) : (Array.isArray(v) ? v : []).map(v => [String(v)])
  const [rows, setRows] = useState(() => decode(value))
  const [query, setQuery] = useState('')
  const sent = useRef(value)
  useEffect(() => { if (value !== sent.current) { setRows(decode(value)); sent.current = value } }, [value])
  const update = (next: string[][]) => {
    setRows(next)
    // JSON objects cannot represent duplicate keys. Keep these rows locally
    // until they can be sent losslessly; Rust validates all representable values.
    if (map && new Set(next.map(r => r[0])).size !== next.length) { onInvalid('Duplicate correction'); return }
    const nextValue = map ? Object.fromEntries(next) : next.map(r => r[0] ?? '')
    sent.current = nextValue
    onValue(nextValue)
  }
  return <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
    <input testId={`search-${field.id}`} placeholder="Filter entries" value={query} onChange={e => setQuery(e.value ?? '')} style={{ ...INPUT_STYLE, width: 320 }} />
    {rows.map((row, i) => row.join(' ').toLowerCase().includes(query.toLowerCase()) ? <div key={i} style={{ display: 'flex', flexDirection: 'row', gap: 8 }}>
      {row.map((text, j) => <input key={j} testId={`entry-${field.id}-${i}-${j}`} value={text} onChange={e => update(rows.map((r, n) => n === i ? r.map((v, k) => k === j ? e.value ?? '' : v) : r))} style={{ ...INPUT_STYLE, width: map ? 220 : 360 }} />)}
      <Button label="Remove" onClick={() => update(rows.filter((_, n) => n !== i))} />
    </div> : null)}
    <Button label={map ? 'Add correction' : 'Add term'} testId={`add-${field.id}`} onClick={() => { setQuery(''); update([...rows, map ? ['', ''] : ['']]) }} />
  </div>
}

function FieldRow({ field, value, error, note, extra, devices, onValue, onInvalid }: {
  field: FieldMeta
  value: Json
  error?: string
  note?: { text: string; bad: boolean }
  extra: Choice[]
  devices: InputDevice[] | null
  onValue: (value: Json) => void
  onInvalid: (message: string) => void
}) {
  let control: ReactNode
  switch (field.kind.type) {
    case 'choice':
      control = (
        <ChoiceControl field={field} value={value} choices={[...field.kind.choices, ...extra]} onValue={onValue} />
      )
      break
    case 'bool':
      control = <Switch field={field} value={value} onValue={onValue} />
      break
    case 'int':
    case 'float':
      control = <NumberControl field={field} value={value} onValue={onValue} onInvalid={onInvalid} />
      break
    case 'text':
      control = <TextControl field={field} value={value} onValue={onValue} />
      break
    case 'optional_text':
      control = <div style={{ display: 'flex', flexDirection: 'row', gap: 8 }}><TextControl field={field} value={value} onValue={onValue} /><Button label="Unset" onClick={() => onValue(null)} /></div>
      break
    case 'string_list':
    case 'string_map':
      control = <CollectionControl field={field} value={value} onValue={onValue} onInvalid={onInvalid} />
      break
    case 'audio_device': {
      const { selected, choices } = deviceChoices(devices, value)
      control = (
        <ChoiceControl
          field={field}
          value={selected}
          choices={choices}
          onValue={(v) => onValue(v === DEFAULT_DEVICE ? null : v.startsWith('name:') ? v.slice(5) : Number(v.slice(6)))}
        />
      )
      break
    }
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 14 }}>
        <Text color={C.dim} size={12}>{field.label}</Text>
        <Button label="Reset to default" testId={`reset-${field.id}`} onClick={() => onValue(field.default ?? null)} />
      </div>
      {control}
      {field.help ? (
        <Text color={C.faint} size={12}>
          {field.help}
        </Text>
      ) : null}
      {note ? (
        <Text color={note.bad ? C.warn : C.faint} size={12} testId={`note-${field.id}`}>
          {note.text}
        </Text>
      ) : null}
      {error ? (
        <Text color={C.bad} size={12} testId={`error-${field.id}`}>
          {error}
        </Text>
      ) : null}
    </div>
  )
}

function ServicePage({ status, snap }: { status: ServiceStatus | null; snap: Snapshot | null }) {
  const rows: [string, string][] = [
    ['Service', status ? status.active_state : '…'],
    ['Overlay', status?.ui_ready === true ? 'Ready' : status?.active_state === 'active' && status.ui_ready === false ? 'Starting' : '—'],
    ['Dictation', status?.stt ?? '—'],
    ['Read aloud', status?.tts ?? '—'],
  ]
  if (snap) rows.push(['Config file', snap.path])
  if (snap?.config_error) rows.push(['Config error', snap.config_error])
  for (const secret of snap?.secrets ?? []) rows.push([secret.env, secret.present ? `Set · ${secret.source ?? 'source unavailable'}` : 'Not set'])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: 'flex', flexDirection: 'row', gap: 12 }}>
          <div style={{ width: 180, flexShrink: 0 }}>
            <Text color={C.dim}>{label}</Text>
          </div>
          <Text testId={`status-${label}`}>{value}</Text>
        </div>
      ))}
    </div>
  )
}

function serviceSummary(status: ServiceStatus | null): { text: string; color: string } {
  if (!status) return { text: 'Service …', color: C.dim }
  if (status.active_state === 'active') {
    return status.ui_ready === false
      ? { text: 'Service starting', color: C.warn }
      : { text: 'Service running', color: C.ok }
  }
  if (status.active_state === 'failed') return { text: 'Service failed', color: C.bad }
  return { text: `Service ${status.active_state}`, color: C.dim }
}

/** Matches the edges of the splash art so it blends into the window. */
const SPLASH_BG = '#050208'
/** Keep the splash up at least this long so it reads as intentional, not a flash. */
const SPLASH_MIN_MS = 1000

function Splash({ src }: { src: string }) {
  return (
    <motion.div
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: SPLASH_BG,
      }}
    >
      <div testId="splash" style={{ width: '80%', height: '50%', maxWidth: 760, maxHeight: 362 }}>
        <img src={src} alt="ShuVoice" objectFit="contain" style={{ width: '100%', height: '100%' }} />
      </div>
      <Text color={C.dim}>Loading settings…</Text>
    </motion.div>
  )
}

export function App({ bridge, brand, onboarding: initialOnboarding = false }: { bridge: Bridge; brand: Brand; onboarding?: boolean }) {
  const [hello, setHello] = useState<{ features?: string[] } | null>(null)
  const [onboarding, setOnboarding] = useState(initialOnboarding)
  const [stage, setStage] = useState(0)
  const [query, setQuery] = useState('')
  const [highlight, setHighlight] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const [outputs, setOutputs] = useState<InputDevice[]>([])
  const [hints, setHints] = useState<{ supported: boolean; detail: string } | null>(null)
  const [previewText, setPreviewText] = useState('')
  const [preview, setPreview] = useState<{ output: string; builtins: Record<string, string> } | null>(null)
  const [models, setModels] = useState<{ id: string; label: string; installed: boolean; size_hint: string | null }[]>([])
  const [download, setDownload] = useState<{ id: string; text: string; fraction: number | null } | null>(null)
  const [operationError, setOperationError] = useState('')
  const [modelEpoch, setModelEpoch] = useState(0)
  const [shortcut, setShortcut] = useState<{ current: { id: string; label: string } | null; options: { id: string; label: string }[]; config_path: string | null; error: string | null } | null>(null)
  const [binding, setBinding] = useState('')
  const [shortcutResult, setShortcutResult] = useState<{ status: string; message: string; conflicts: string[]; backup: string | null } | null>(null)
  const [shortcutPreview, setShortcutPreview] = useState<string | null>(null)
  const [formEpoch, setFormEpoch] = useState(0)
  const [shortcutBusy, setShortcutBusy] = useState(false)
  const feature = (name: string) => supports(hello, name)
  const [schema, setSchema] = useState<Schema | null>(null)
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [draft, setDraft] = useState<Values>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [devices, setDevices] = useState<InputDevice[] | null>(null)
  const [status, setStatus] = useState<ServiceStatus | null>(null)
  const [page, setPage] = useState<Page>('speech')
  const [phase, setPhase] = useState<Phase>({ kind: 'loading' })
  // State updates land after the event; a ref blocks a second click in the same tick.
  const applying = useRef(false)
  const [splashHeld, setSplashHeld] = useState(true)
  useEffect(() => {
    const timer = setTimeout(() => setSplashHeld(false), SPLASH_MIN_MS)
    return () => clearTimeout(timer)
  }, [])

  const load = useCallback(async () => {
    const [nextSchema, nextSnap, greeting] = await Promise.all([
      bridge.call<Schema>('schema'),
      bridge.call<Snapshot>('snapshot'),
      bridge.call<{ features?: string[] }>('hello'),
    ])
    setSchema(nextSchema)
    setSnap(nextSnap)
    setHello(greeting)
    setDraft(initialOnboarding && supports(greeting, 'onboarding_defaults') ? (await bridge.call<{ values: Values }>('onboarding_defaults')).values : nextSnap.values)
    setErrors({})
    setPhase({ kind: 'idle' })
  }, [bridge])

  useEffect(() => {
    load().catch((e: Error) => { setHello(null); setOnboarding(true); setPhase({ kind: 'error', message: e.message }) })
    bridge
      .call<{ devices: InputDevice[]; outputs?: InputDevice[] }>('devices')
      .then((r) => { setDevices(r.devices); setOutputs(r.outputs ?? []) })
      .catch(() => setDevices([]))
  }, [load, bridge])

  useEffect(() => {
    const tick = () => bridge.call<ServiceStatus>('status').then(setStatus).catch(() => {})
    tick()
    const timer = setInterval(tick, 3000)
    return () => clearInterval(timer)
  }, [bridge])

  const changes = useMemo(() => (snap ? diff(snap.values, draft) : {}), [snap, draft])
  useEffect(() => {
    let stale = false
    const timer = setTimeout(() => {
      if (supports(hello, 'capabilities')) void bridge.call<{ vocabulary_hints: { supported: boolean; detail: string } }>('capabilities', { changes }).then(r => { if (!stale) setHints(r.vocabulary_hints) }).catch(e => { if (!stale) setOperationError(e.message) })
      if (supports(hello, 'models')) void bridge.call<{ required: typeof models }>('models', { changes }).then(r => { if (!stale) setModels(r.required) }).catch(e => { if (!stale) setOperationError(e.message) })
      if (supports(hello, 'corrections_preview')) void bridge.call<{ output: string; builtins: Record<string, string> }>('corrections_preview', { changes, text: previewText }).then(r => { if (!stale) setPreview(r) }).catch(e => { if (!stale) setOperationError(e.message) })
    }, 250)
    return () => { stale = true; clearTimeout(timer) }
  }, [hello, changes, previewText, modelEpoch])
  useEffect(() => {
    if (supports(hello, 'shortcut_get')) void bridge.call<NonNullable<typeof shortcut>>('shortcut_get').then(r => { setShortcut(r); setBinding(r.current?.id ?? r.options[0]?.id ?? '') }).catch(e => setOperationError(e.message))
  }, [hello])
  const downloadModel = async (id: string) => {
    if (applying.current) return
    applying.current = true
    setOperationError('')
    setDownload({ id, text: 'Downloading…', fraction: null })
    try { await bridge.call('model_download', { id }, e => setDownload({ id, text: String(e.text ?? 'Downloading…'), fraction: typeof e.fraction === 'number' ? e.fraction : null })); setModelEpoch(n => n + 1) }
    catch (e) { if (!(e instanceof BridgeError && e.kind === 'cancelled')) setOperationError((e as Error).message) }
    finally { setDownload(null); applying.current = false }
  }
  const setShortcutBinding = async (dry_run: boolean) => {
    if (shortcutBusy) return
    setShortcutBusy(true)
    try { const r = await bridge.call<NonNullable<typeof shortcutResult>>('shortcut_set', { id: binding, dry_run }); setShortcutResult(r); setShortcutPreview(dry_run && !['error', 'unsupported'].includes(r.status) ? binding : null); if (!dry_run) setShortcut(await bridge.call<NonNullable<typeof shortcut>>('shortcut_get')) }
    catch (e) { setOperationError((e as Error).message); setShortcutPreview(null) }
    finally { setShortcutBusy(false) }
  }
  const dirty = Object.keys(changes).length
  const hasErrors = Object.keys(errors).length > 0
  const busy = phase.kind === 'applying' || phase.kind === 'loading' || download !== null

  const setValue = (id: string, value: Json) => {
    if (busy) return
    setDraft((d) => ({ ...d, [id]: value }))
    setErrors(({ [id]: _, '': __, ...rest }) => rest)
    if (phase.kind === 'done' || phase.kind === 'error') setPhase({ kind: 'idle' })
  }

  const showErrors = (list: FieldError[]) => {
    setErrors(errorsByField(list))
    const first = schema?.fields.find((f) => list.some((e) => e.field === f.id))
    if (first) setPage(first.section)
    if (first) { setHighlight(first.id); setAdvanced(true) }
  }

  const apply = async () => {
    if (!snap || (!onboarding && dirty === 0) || hasErrors || busy || applying.current) return
    applying.current = true
    setPhase({ kind: 'applying', step: 'validating', busy: [] })
    const onEvent = (event: BridgeEvent) => {
      if (event.event === 'progress' && event.phase) {
        setPhase({ kind: 'applying', step: event.phase as Step, busy: event.busy ?? [] })
      }
    }
    try {
      const result = await bridge.call<ApplyResult>('apply', { revision: snap.revision, changes, ...(onboarding ? { onboarding: true } : {}) }, onEvent)
      const fresh = await bridge.call<Snapshot>('snapshot')
      setSnap(fresh)
      setDraft(fresh.values)
      setPhase(outcomePhase(result))
      if (onboarding && result.restart.outcome === 'ready') setOnboarding(false)
    } catch (e) {
      if (!(e instanceof BridgeError)) {
        setPhase({ kind: 'error', message: (e as Error).message })
      } else if (e.kind === 'conflict') {
        setPhase({ kind: 'conflict' })
      } else if (e.kind === 'invalid') {
        showErrors((e.data.errors as FieldError[]) ?? [])
        setPhase({ kind: 'idle' })
      } else if (e.kind === 'cancelled') {
        setPhase({ kind: 'idle' })
      } else {
        setPhase({ kind: 'error', message: e.message })
      }
    } finally {
      applying.current = false
    }
  }

  const cancel = () => {
    void bridge.call('cancel').catch(() => {})
  }

  const revert = () => {
    if (!snap) return
    setFormEpoch(n => n + 1)
    setDraft(snap.values)
    setErrors({})
    setPhase({ kind: 'idle' })
  }

  const fields = (schema?.fields ?? []).filter((f) => f.section === page && (visible(f, draft) || f.id === highlight))
  const pages: Page[] = [...new Set<Section>([...(schema?.sections ?? []), 'vocabulary', 'advanced']), 'shortcuts', 'service']
  const summary = serviceSummary(status)

  let footerText = ''
  let footerColor = C.dim
  if (phase.kind === 'loading') footerText = 'Loading…'
  else if (phase.kind === 'applying') {
    footerText =
      phase.step === 'waiting_idle'
        ? busyText(phase.busy)
        : phase.step === 'restarting'
          ? 'Restarting ShuVoice…'
          : phase.step === 'reserving' ? 'Reserving ShuVoice…' : phase.step === 'validating' ? 'Checking settings…' : 'Saving…'
    footerColor = phase.step === 'waiting_idle' ? C.warn : C.dim
  } else if (phase.kind === 'done') [footerText, footerColor] = [phase.text, phase.tone === 'ok' ? C.ok : C.warn]
  else if (phase.kind === 'conflict') [footerText, footerColor] = ['Config changed on disk', C.warn]
  else if (phase.kind === 'error') [footerText, footerColor] = [phase.message, C.bad]
  else if (errors['']) [footerText, footerColor] = [errors[''], C.bad]
  else if (dirty > 0) [footerText, footerColor] = [dirty === 1 ? '1 unsaved change' : `${dirty} unsaved changes`, C.warn]

  const keyNote = (field: FieldMeta): { text: string; bad: boolean } | undefined => {
    const value = draft[field.id]
    const secret =
      field.id === 'asr.asr_backend' && value === 'openai_realtime'
        ? snap?.secrets.find((s) => s.used_by === 'asr.asr_backend:openai_realtime')
        : field.id === 'tts.tts_backend' && (value === 'elevenlabs' || value === 'openai')
          ? snap?.secrets.find((s) => s.used_by === 'tts.tts_backend')
          : undefined
    if (!secret) return undefined
    return secret.present
      ? { text: `${secret.env} is set`, bad: false }
      : { text: `${secret.env} is not set — add it to ~/.config/shuvoice/local.dev`, bad: true }
  }

  const showSplash = splashHeld || phase.kind === 'loading'
  const stages = ['Engine', 'Microphone', 'Read-aloud', 'Shortcut', 'Finish']
  const moveStage = (next: number) => { setStage(next); setPage((['speech', 'audio', 'text_to_speech', 'shortcuts', 'service'] as Page[])[next]!); setHighlight('') }
  const row = (f: FieldMeta) => <div key={`${f.id}-${formEpoch}`} style={{ padding: 8, flexShrink: 0, borderRadius: 6, borderWidth: f.id === highlight ? 1 : 0, borderColor: C.accent }}><FieldRow field={f} value={draft[f.id] ?? null} error={errors[f.id]} note={keyNote(f)} extra={snap?.extra_choices[f.id] ?? []} devices={f.kind.type === 'audio_device' && f.kind.direction === 'output' ? outputs : devices} onValue={v => setValue(f.id, v)} onInvalid={message => setErrors(e => ({ ...e, [f.id]: message }))} /></div>

  return (
    <div style={{ position: 'relative', display: 'flex', flexDirection: 'row', width: '100%', height: '100%', backgroundColor: C.bg }}>
      <div style={{ width: 200, flexShrink: 0, backgroundColor: C.side, padding: 12, gap: 4, display: 'flex', flexDirection: 'column' }}>
        <div testId="sidebar-logo" style={{ paddingLeft: 4, paddingRight: 4, paddingTop: 4, paddingBottom: 12 }}>
          <img src={brand.lockup} alt="ShuVoice" objectFit="contain" style={{ width: 168, height: 73 }} />
        </div>
        {onboarding ? <><Text bold>Set up ShuVoice</Text>{stages.map((label, i) => <Button key={label} label={label} primary={i === stage} onClick={() => moveStage(i)} />)}<Button label="Skip to settings" testId="skip-onboarding" onClick={() => setOnboarding(false)} /></> : pages.map((p) => {
          const label = p === 'service' ? 'Service' : p === 'shortcuts' ? 'Shortcuts' : SECTION_LABELS[p]
          const pageHasError = p !== 'service' && schema?.fields.some((f) => f.section === p && errors[f.id])
          return (
            <div
              key={p}
              testId={`nav-${p}`}
              tabIndex={0}
              onKeyDown={e => { if (e.key === 'enter' || e.key === 'space') setPage(p) }}
              onClick={() => { setPage(p); setHighlight(''); setAdvanced(false) }}
              style={{
                padding: 8,
                borderRadius: 6,
                cursor: 'pointer',
                backgroundColor: p === page ? C.card : C.side,
                hover: { backgroundColor: C.card },
              }}
            >
              <Text color={pageHasError ? C.bad : p === page ? C.text : C.dim}>{label}</Text>
            </div>
          )
        })}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minWidth: 0 }}>
        {!onboarding ? <input testId="global-search" placeholder="Search settings" value={query} onChange={e => setQuery(e.value ?? '')} style={{ ...INPUT_STYLE, margin: 16 }} /> : null}
        <div key={`${page}-${highlight}`} style={{ flexGrow: 1, minHeight: 0, overflowY: 'scroll', padding: 24, gap: 20, display: 'flex', flexDirection: 'column' }}>
          {searchFields(schema?.fields ?? [], query).map(f => <Button key={f.id} label={`${SECTION_LABELS[f.section]} · ${f.label}`} testId={`result-${f.id}`} onClick={() => { setPage(f.section); setHighlight(f.id); setAdvanced(true); setQuery('') }} />)}
          <Text bold size={20} testId="page-title">
            {onboarding ? stages[stage] : page === 'service' ? 'Service' : page === 'shortcuts' ? 'Shortcuts' : SECTION_LABELS[page]}
          </Text>
          {operationError ? <Text color={C.bad}>{operationError}</Text> : null}
          {page === 'advanced' && fields.length === 0 ? <Text color={C.dim}>No advanced fields exposed by this bridge.</Text> : null}
          {onboarding && !feature('onboarding_defaults') ? <Text color={C.warn}>Setup requires a newer ShuVoice bridge. Open settings or run shuvoice wizard.</Text> : null}
          {page === 'speech' && feature('models') ? <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {models.map(m => <div key={m.id} style={{ display: 'flex', flexDirection: 'row', gap: 12, alignItems: 'center' }}><Text testId={`model-${m.id}`}>{`${m.label} · ${m.installed ? 'Installed' : m.size_hint ?? 'Not installed'}`}</Text>{!m.installed && feature('model_download') ? <Button label="Download" testId={`download-${m.id}`} disabled={busy} onClick={() => void downloadModel(m.id)} /> : null}</div>)}
          </div> : null}
          {download ? <><Text>{download.text}</Text><div style={{ height: 6, backgroundColor: C.line }}><div style={{ height: 6, width: `${Math.max(0, Math.min(1, download.fraction ?? 0.1)) * 100}%`, backgroundColor: C.accent }} /></div><Button label="Cancel download" onClick={cancel} /></> : null}
          {page === 'vocabulary' ? <>
            {hints ? <><Text color={hints.supported ? C.ok : C.warn}>{hints.supported ? 'Hints supported' : 'Hints unsupported'}</Text><Text color={C.dim}>{hints.detail}</Text></> : <Text color={C.dim}>Hint support unavailable from this bridge.</Text>}
            {feature('corrections_preview') ? <><Text bold>Correction preview</Text><input testId="correction-preview" value={previewText} placeholder="Try a transcription" onChange={e => setPreviewText(e.value ?? '')} style={INPUT_STYLE} /><Text>{preview?.output ?? ''}</Text><Text color={C.dim}>{`Built-ins: ${Object.entries(preview?.builtins ?? {}).map(([a, b]) => `${a} → ${b}`).join(' · ')}`}</Text></> : null}
          </> : null}
          {page === 'shortcuts' ? <>
            {!feature('shortcut_get') || !feature('shortcut_set') ? <Text color={C.dim}>Shortcut editing requires a newer ShuVoice bridge.</Text> : <>
              <Text>{`Push-to-talk: ${shortcut?.current?.label ?? 'Not configured'}`}</Text>
              {shortcut?.error ? <Text color={C.warn}>{shortcut.error}</Text> : null}
              {shortcut?.config_path ? <Text color={C.dim}>{shortcut.config_path}</Text> : null}
              <ChoiceControl field={{ id: 'shortcut', section: 'advanced', label: 'Shortcut', help: '', unit: '', kind: { type: 'choice', choices: [] } }} value={binding} choices={(shortcut?.options ?? []).map(o => ({ value: o.id, label: o.label }))} onValue={v => { setBinding(v); setShortcutPreview(null); setShortcutResult(null) }} />
              <Text color={C.dim}>Applies immediately to Hyprland.</Text>
              <Button label="Preview shortcut" testId="shortcut-preview" disabled={!binding || !!shortcut?.error || shortcutBusy} onClick={() => void setShortcutBinding(true)} />
              {shortcutResult ? <><Text>{shortcutResult.message}</Text>{shortcutResult.conflicts.map(c => <Text key={c} color={C.warn}>{c}</Text>)}{shortcutResult.backup ? <Text>{`Backup: ${shortcutResult.backup}`}</Text> : null}</> : null}
              {shortcutPreview === binding ? <><Button label="Confirm shortcut" testId="shortcut-confirm" disabled={shortcutBusy} onClick={() => void setShortcutBinding(false)} /><Button label="Cancel" onClick={() => setShortcutPreview(null)} /></> : null}
            </>}
          </> : null}
          {snap?.config_error ? (
            <div style={{ padding: 10, borderRadius: 6, backgroundColor: '#3a2228' }}>
              <Text color={C.bad} testId="config-error">{`Current config is invalid: ${snap.config_error}`}</Text>
            </div>
          ) : null}
          {page === 'service' ? (
            <ServicePage status={status} snap={snap} />
          ) : (
            <>{highlight ? fields.filter(f => f.id === highlight).map(row) : null}{fields.filter(f => !f.advanced && f.id !== highlight).map(row)}{fields.some(f => f.advanced) ? <Button label={advanced ? 'Hide advanced' : 'Advanced'} testId="advanced-toggle" onClick={() => setAdvanced(!advanced)} /> : null}{advanced ? fields.filter(f => f.advanced && f.id !== highlight).map(row) : null}</>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 12, padding: 14, borderTopWidth: 1, borderColor: C.line, backgroundColor: C.side }}>
          <div style={{ flexGrow: 1, minWidth: 0, display: 'flex', flexDirection: 'row', gap: 12, alignItems: 'center' }}>
            <Text color={footerColor} testId="footer-status">
              {footerText}
            </Text>
            {phase.kind === 'conflict' ? <Button label="Reload" onClick={() => void load()} testId="reload" /> : null}
            {phase.kind === 'applying' && phase.step === 'waiting_idle' ? (
              <Button label="Cancel" onClick={cancel} testId="cancel" />
            ) : null}
          </div>
          <Text color={summary.color} testId="service-summary">
            {summary.text}
          </Text>
          {models.some(m => !m.installed) ? <Text color={C.warn}>Required model missing</Text> : null}
          {onboarding && stage > 0 ? <Button label="Back" disabled={busy} onClick={() => moveStage(stage - 1)} /> : null}
          {onboarding && stage < 4 ? <Button label="Next" testId="onboarding-next" primary disabled={busy} onClick={() => moveStage(stage + 1)} /> : null}
          {(dirty > 0 || hasErrors) && !busy ? <Button label="Revert" onClick={revert} testId="revert" /> : null}
          <Button
            label={onboarding ? 'Finish setup' : 'Apply & Restart'}
            primary
            disabled={(!onboarding && dirty === 0) || hasErrors || busy || (onboarding && (stage !== 4 || !feature('onboarding_defaults')))}
            onClick={() => void apply()}
            testId="apply"
          />
        </div>
      </div>
      <AnimatePresence>{showSplash ? <Splash key="splash" src={brand.splash} /> : null}</AnimatePresence>
    </div>
  )
}
