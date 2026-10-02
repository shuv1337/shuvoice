import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { type Bridge, BridgeError, type BridgeEvent } from './bridge.ts'
import { diff, errorsByField, supports, type Values } from './draft.ts'
import type {
  ApplyResult,
  FieldError,
  InputDevice,
  Json,
  Schema,
  ServiceStatus,
  Snapshot,
} from './schema.ts'
import { SPLASH_MIN_MS } from './components/splash.tsx'
import { outcomePhase, progressPhase, type Page, type Phase } from './apply-state.ts'

export function useSettings(bridge: Bridge, initialOnboarding = false) {
  const [hello, setHello] = useState<{ features?: string[] } | null>(null)
  const [onboarding, setOnboarding] = useState(initialOnboarding)
  const [stage, setStage] = useState(0)
  const [query, setQuery] = useState('')
  const [highlight, setHighlight] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const [outputs, setOutputs] = useState<InputDevice[]>([])
  const [hints, setHints] = useState<{ supported: boolean; detail: string } | null>(null)
  const [previewText, setPreviewText] = useState('')
  const [preview, setPreview] = useState<{
    output: string
    builtins: Record<string, string>
  } | null>(null)
  const [models, setModels] = useState<
    { id: string; label: string; installed: boolean; size_hint: string | null }[]
  >([])
  const [download, setDownload] = useState<{
    id: string
    text: string
    fraction: number | null
  } | null>(null)
  const [operationError, setOperationError] = useState('')
  const [modelEpoch, setModelEpoch] = useState(0)
  const [shortcut, setShortcut] = useState<{
    current: { id: string; label: string } | null
    options: { id: string; label: string }[]
    config_path: string | null
    error: string | null
  } | null>(null)
  const [binding, setBinding] = useState('')
  const [shortcutResult, setShortcutResult] = useState<{
    status: string
    message: string
    conflicts: string[]
    backup: string | null
  } | null>(null)
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
    setDraft(
      initialOnboarding && supports(greeting, 'onboarding_defaults')
        ? (await bridge.call<{ values: Values }>('onboarding_defaults')).values
        : nextSnap.values,
    )
    setErrors({})
    setPhase({ kind: 'idle' })
  }, [bridge])

  useEffect(() => {
    load().catch((e: Error) => {
      setHello(null)
      setOnboarding(true)
      setPhase({ kind: 'error', message: e.message })
    })
    bridge
      .call<{ devices: InputDevice[]; outputs?: InputDevice[] }>('devices')
      .then((r) => {
        setDevices(r.devices)
        setOutputs(r.outputs ?? [])
      })
      .catch(() => setDevices([]))
  }, [load, bridge])

  useEffect(() => {
    const tick = () =>
      bridge
        .call<ServiceStatus>('status')
        .then(setStatus)
        .catch(() => {})
    tick()
    const timer = setInterval(tick, 3000)
    return () => clearInterval(timer)
  }, [bridge])

  const changes = useMemo(() => (snap ? diff(snap.values, draft) : {}), [snap, draft])
  useEffect(() => {
    let stale = false
    // A previous draft's validation failure must not survive a corrected draft.
    setOperationError('')
    const timer = setTimeout(() => {
      if (supports(hello, 'capabilities'))
        void bridge
          .call<{ vocabulary_hints: { supported: boolean; detail: string } }>('capabilities', {
            changes,
          })
          .then((r) => {
            if (!stale) setHints(r.vocabulary_hints)
          })
          .catch((e) => {
            if (!stale) setOperationError(e.message)
          })
      if (supports(hello, 'models'))
        void bridge
          .call<{ required: typeof models }>('models', { changes })
          .then((r) => {
            if (!stale) setModels(r.required)
          })
          .catch((e) => {
            if (!stale) setOperationError(e.message)
          })
      if (supports(hello, 'corrections_preview'))
        void bridge
          .call<{ output: string; builtins: Record<string, string> }>('corrections_preview', {
            changes,
            text: previewText,
          })
          .then((r) => {
            if (!stale) setPreview(r)
          })
          .catch((e) => {
            if (!stale) setOperationError(e.message)
          })
    }, 250)
    return () => {
      stale = true
      clearTimeout(timer)
    }
  }, [hello, changes, previewText, modelEpoch])
  useEffect(() => {
    if (supports(hello, 'shortcut_get'))
      void bridge
        .call<NonNullable<typeof shortcut>>('shortcut_get')
        .then((r) => {
          setShortcut(r)
          setBinding(r.current?.id ?? r.options[0]?.id ?? '')
        })
        .catch((e) => setOperationError(e.message))
  }, [hello])
  const downloadModel = async (id: string) => {
    if (applying.current) return
    applying.current = true
    setOperationError('')
    setDownload({ id, text: 'Downloading…', fraction: null })
    try {
      await bridge.call('model_download', { id }, (e) =>
        setDownload({
          id,
          text: String(e.text ?? 'Downloading…'),
          fraction: typeof e.fraction === 'number' ? e.fraction : null,
        }),
      )
      setModelEpoch((n) => n + 1)
    } catch (e) {
      if (!(e instanceof BridgeError && e.kind === 'cancelled'))
        setOperationError((e as Error).message)
    } finally {
      setDownload(null)
      applying.current = false
    }
  }
  const setShortcutBinding = async (dry_run: boolean) => {
    if (shortcutBusy) return
    setShortcutBusy(true)
    try {
      const r = await bridge.call<NonNullable<typeof shortcutResult>>('shortcut_set', {
        id: binding,
        dry_run,
      })
      setShortcutResult(r)
      setShortcutPreview(dry_run && !['error', 'unsupported'].includes(r.status) ? binding : null)
      if (!dry_run) setShortcut(await bridge.call<NonNullable<typeof shortcut>>('shortcut_get'))
    } catch (e) {
      setOperationError((e as Error).message)
      setShortcutPreview(null)
    } finally {
      setShortcutBusy(false)
    }
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
    if (first) {
      setHighlight(first.id)
      setAdvanced(true)
    }
  }

  const apply = async () => {
    if (!snap || (!onboarding && dirty === 0) || hasErrors || busy || applying.current) return
    applying.current = true
    setPhase({ kind: 'applying', step: 'validating', busy: [] })
    const onEvent = (event: BridgeEvent) => {
      if (event.event === 'progress' && event.phase) {
        setPhase((previous) => progressPhase(event, previous))
      }
    }
    try {
      const result = await bridge.call<ApplyResult>(
        'apply',
        { revision: snap.revision, changes, ...(onboarding ? { onboarding: true } : {}) },
        onEvent,
      )
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
    setFormEpoch((n) => n + 1)
    setDraft(snap.values)
    setErrors({})
    setPhase({ kind: 'idle' })
  }

  return {
    hello,
    onboarding,
    setOnboarding,
    stage,
    setStage,
    query,
    setQuery,
    highlight,
    setHighlight,
    advanced,
    setAdvanced,
    outputs,
    hints,
    previewText,
    setPreviewText,
    preview,
    models,
    download,
    operationError,
    shortcut,
    binding,
    setBinding,
    shortcutResult,
    setShortcutResult,
    shortcutPreview,
    setShortcutPreview,
    formEpoch,
    shortcutBusy,
    feature,
    schema,
    snap,
    draft,
    errors,
    setErrors,
    devices,
    status,
    page,
    setPage,
    phase,
    splashHeld,
    load,
    changes,
    downloadModel,
    setShortcutBinding,
    dirty,
    hasErrors,
    busy,
    setValue,
    apply,
    cancel,
    revert,
  }
}

export type SettingsState = ReturnType<typeof useSettings>
