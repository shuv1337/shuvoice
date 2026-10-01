import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@gpuix/react/select'
import { type Bridge, BridgeError, type BridgeEvent } from './bridge.ts'
import { diff, errorsByField, formatNumber, parseNumber, step, visible, type Values } from './draft.ts'
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
  line: '#2f2f38',
  text: '#e6e6ea',
  dim: '#9a9aa6',
  faint: '#6d6d78',
  accent: '#7aa2f7',
  onAccent: '#101014',
  ok: '#9ece6a',
  warn: '#e0af68',
  bad: '#f7768e',
}

type Page = Section | 'service'

type Step = 'validating' | 'waiting_idle' | 'saving' | 'restarting'

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
        style={{ padding: 8, borderRadius: 6, backgroundColor: C.card, width: 320 }}
      >
        <SelectValue style={{ color: C.text }} placeholder={<Text color={C.dim}>Choose…</Text>} />
      </SelectTrigger>
      <SelectContent style={{ backgroundColor: C.card, borderRadius: 6, padding: 4 }}>
        {choices.map((c) => (
          <SelectItem
            key={c.value}
            value={c.value}
            testId={`option-${field.id}-${c.value}`}
            style={({ highlighted }) => ({
              padding: 6,
              borderRadius: 4,
              backgroundColor: highlighted ? C.line : C.card,
            })}
          >
            <Text>{c.label}</Text>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function Switch({ field, value, onValue }: { field: FieldMeta; value: Json; onValue: (value: boolean) => void }) {
  const on = value === true
  return (
    <div
      testId={`field-${field.id}`}
      onClick={() => onValue(!on)}
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
        style={{ padding: 8, borderRadius: 6, backgroundColor: C.card, color: C.text, width: 96 }}
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
      style={{ padding: 8, borderRadius: 6, backgroundColor: C.card, color: C.text, width: 420 }}
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
      <Text color={C.dim} size={12}>
        {field.label}
      </Text>
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
    ['Overlay', status?.ui_ready === true ? 'Ready' : status?.ui_ready === false ? 'Starting' : '—'],
    ['Dictation', status?.stt ?? '—'],
    ['Read aloud', status?.tts ?? '—'],
  ]
  if (snap) rows.push(['Config file', snap.path])
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

export function App({ bridge }: { bridge: Bridge }) {
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

  const load = useCallback(async () => {
    const [nextSchema, nextSnap] = await Promise.all([
      bridge.call<Schema>('schema'),
      bridge.call<Snapshot>('snapshot'),
    ])
    setSchema(nextSchema)
    setSnap(nextSnap)
    setDraft(nextSnap.values)
    setErrors({})
    setPhase({ kind: 'idle' })
  }, [bridge])

  useEffect(() => {
    load().catch((e: Error) => setPhase({ kind: 'error', message: e.message }))
    bridge
      .call<{ devices: InputDevice[] }>('devices')
      .then((r) => setDevices(r.devices))
      .catch(() => setDevices([]))
  }, [load, bridge])

  useEffect(() => {
    const tick = () => bridge.call<ServiceStatus>('status').then(setStatus).catch(() => {})
    tick()
    const timer = setInterval(tick, 3000)
    return () => clearInterval(timer)
  }, [bridge])

  const changes = useMemo(() => (snap ? diff(snap.values, draft) : {}), [snap, draft])
  const dirty = Object.keys(changes).length
  const hasErrors = Object.keys(errors).length > 0
  const busy = phase.kind === 'applying' || phase.kind === 'loading'

  const setValue = (id: string, value: Json) => {
    setDraft((d) => ({ ...d, [id]: value }))
    setErrors(({ [id]: _, '': __, ...rest }) => rest)
    if (phase.kind === 'done' || phase.kind === 'error') setPhase({ kind: 'idle' })
  }

  const showErrors = (list: FieldError[]) => {
    setErrors(errorsByField(list))
    const first = schema?.fields.find((f) => list.some((e) => e.field === f.id))
    if (first) setPage(first.section)
  }

  const apply = async () => {
    if (!snap || dirty === 0 || hasErrors || busy || applying.current) return
    applying.current = true
    setPhase({ kind: 'applying', step: 'validating', busy: [] })
    const onEvent = (event: BridgeEvent) => {
      if (event.event === 'progress' && event.phase) {
        setPhase({ kind: 'applying', step: event.phase as Step, busy: event.busy ?? [] })
      }
    }
    try {
      const result = await bridge.call<ApplyResult>('apply', { revision: snap.revision, changes }, onEvent)
      const fresh = await bridge.call<Snapshot>('snapshot')
      setSnap(fresh)
      setDraft(fresh.values)
      setPhase(outcomePhase(result))
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
    setDraft(snap.values)
    setErrors({})
    setPhase({ kind: 'idle' })
  }

  const fields = (schema?.fields ?? []).filter((f) => f.section === page && visible(f, draft))
  const pages: Page[] = [...(schema?.sections ?? []), 'service']
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
          : 'Saving…'
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

  return (
    <div style={{ display: 'flex', flexDirection: 'row', width: '100%', height: '100%', backgroundColor: C.bg }}>
      <div style={{ width: 200, flexShrink: 0, backgroundColor: C.side, padding: 12, gap: 4, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: 8 }}>
          <Text bold size={15}>
            ShuVoice
          </Text>
        </div>
        {pages.map((p) => {
          const label = p === 'service' ? 'Service' : SECTION_LABELS[p]
          const pageHasError = p !== 'service' && schema?.fields.some((f) => f.section === p && errors[f.id])
          return (
            <div
              key={p}
              testId={`nav-${p}`}
              onClick={() => setPage(p)}
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
        <div style={{ flexGrow: 1, minHeight: 0, overflowY: 'scroll', padding: 24, gap: 20, display: 'flex', flexDirection: 'column' }}>
          <Text bold size={20} testId="page-title">
            {page === 'service' ? 'Service' : SECTION_LABELS[page]}
          </Text>
          {snap?.config_error ? (
            <div style={{ padding: 10, borderRadius: 6, backgroundColor: '#3a2228' }}>
              <Text color={C.bad} testId="config-error">{`Current config is invalid: ${snap.config_error}`}</Text>
            </div>
          ) : null}
          {page === 'service' ? (
            <ServicePage status={status} snap={snap} />
          ) : (
            fields.map((f) => (
              <FieldRow
                key={f.id}
                field={f}
                value={draft[f.id] ?? null}
                error={errors[f.id]}
                note={keyNote(f)}
                extra={snap?.extra_choices[f.id] ?? []}
                devices={devices}
                onValue={(v) => setValue(f.id, v)}
                onInvalid={(message) => setErrors((e) => ({ ...e, [f.id]: message }))}
              />
            ))
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
          {dirty > 0 && !busy ? <Button label="Revert" onClick={revert} testId="revert" /> : null}
          <Button
            label="Apply & Restart"
            primary
            disabled={dirty === 0 || hasErrors || busy}
            onClick={() => void apply()}
            testId="apply"
          />
        </div>
      </div>
    </div>
  )
}
