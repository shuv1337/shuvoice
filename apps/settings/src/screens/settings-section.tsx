import { C } from '../theme.ts'
import { Text, Button, FieldRow } from '../components/controls.tsx'
import { visible } from '../draft.ts'
import type { FieldMeta } from '../schema.ts'
import type { SettingsState } from '../use-settings.ts'

export function SettingsSection({
  schema,
  page,
  draft,
  highlight,
  snap,
  formEpoch,
  errors,
  outputs,
  devices,
  setValue,
  setErrors,
  advanced,
  setAdvanced,
}: SettingsState) {
  const fields = (schema?.fields ?? []).filter(
    (f) => f.section === page && (visible(f, draft) || f.id === highlight),
  )
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

  const row = (f: FieldMeta) => (
    <div
      key={`${f.id}-${formEpoch}`}
      style={{
        padding: 8,
        flexShrink: 0,
        borderRadius: 6,
        borderWidth: f.id === highlight ? 1 : 0,
        borderColor: C.accent,
      }}
    >
      <FieldRow
        field={f}
        value={draft[f.id] ?? null}
        error={errors[f.id]}
        note={keyNote(f)}
        extra={snap?.extra_choices[f.id] ?? []}
        devices={
          f.kind.type === 'audio_device' && f.kind.direction === 'output' ? outputs : devices
        }
        onValue={(v) => setValue(f.id, v)}
        onInvalid={(message) => setErrors((e) => ({ ...e, [f.id]: message }))}
      />
    </div>
  )
  return (
    <>
      {page === 'advanced' && fields.length === 0 ? (
        <Text color={C.dim}>No advanced fields exposed by this bridge.</Text>
      ) : null}
      <>
        {highlight ? fields.filter((f) => f.id === highlight).map(row) : null}
        {fields.filter((f) => !f.advanced && f.id !== highlight).map(row)}
        {fields.some((f) => f.advanced) ? (
          <Button
            label={advanced ? 'Hide advanced' : 'Advanced'}
            testId="advanced-toggle"
            onClick={() => setAdvanced(!advanced)}
          />
        ) : null}
        {advanced ? fields.filter((f) => f.advanced && f.id !== highlight).map(row) : null}
      </>
    </>
  )
}
