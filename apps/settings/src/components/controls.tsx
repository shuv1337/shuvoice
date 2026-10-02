import { Text, Button } from './button.tsx'
import { ChoiceControl } from './choice.tsx'
import { CollectionControl } from './collection.tsx'
export { Text, Button } from './button.tsx'
export { ChoiceControl } from './choice.tsx'
import { useEffect, useState, type ReactNode } from 'react'
import { C, INPUT_STYLE } from '../theme.ts'
import { equal, formatNumber, parseNumber, step } from '../draft.ts'
import type { Choice, FieldMeta, InputDevice, Json } from '../schema.ts'

const DEFAULT_DEVICE = '__default__'

function Switch({
  field,
  value,
  onValue,
}: {
  field: FieldMeta
  value: Json
  onValue: (value: boolean) => void
}) {
  const on = value === true
  return (
    <div
      testId={`field-${field.id}`}
      onClick={() => onValue(!on)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'enter' || e.key === 'space') onValue(!on)
      }}
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
      <div
        style={{ width: 18, height: 18, borderRadius: 9, backgroundColor: on ? C.onAccent : C.dim }}
      />
    </div>
  )
}

function NumberControl({
  field,
  value,
  onValue,
  onInvalid,
}: {
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

function TextControl({
  field,
  value,
  onValue,
}: {
  field: FieldMeta
  value: Json
  onValue: (value: string) => void
}) {
  return (
    <input
      testId={`field-${field.id}`}
      value={typeof value === 'string' ? value : ''}
      onChange={(e) => onValue(e.value ?? '')}
      style={{ ...INPUT_STYLE, width: 420 }}
    />
  )
}

function deviceChoices(
  devices: InputDevice[] | null,
  value: Json,
): { selected: string; choices: Choice[] } {
  const choices: Choice[] = [{ value: DEFAULT_DEVICE, label: 'System default' }]
  for (const d of devices ?? []) choices.push({ value: `name:${d.name}`, label: d.name })
  let selected = DEFAULT_DEVICE
  if (typeof value === 'string') {
    selected = `name:${value}`
    if (!choices.some((c) => c.value === selected))
      choices.push({ value: selected, label: `${value} (not found)` })
  } else if (typeof value === 'number') {
    const match = devices?.find((d) => d.index === value)
    selected = match ? `name:${match.name}` : `index:${value}`
    if (!match) choices.push({ value: selected, label: `Device #${value} (not found)` })
  }
  return { selected, choices }
}

export function FieldRow({
  field,
  value,
  error,
  note,
  extra,
  devices,
  onValue,
  onInvalid,
}: {
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
        <ChoiceControl
          field={field}
          value={value}
          choices={[...field.kind.choices, ...extra]}
          onValue={onValue}
        />
      )
      break
    case 'bool':
      control = <Switch field={field} value={value} onValue={onValue} />
      break
    case 'int':
    case 'float':
      control = (
        <NumberControl field={field} value={value} onValue={onValue} onInvalid={onInvalid} />
      )
      break
    case 'text':
      control = <TextControl field={field} value={value} onValue={onValue} />
      break
    case 'optional_text':
      control = (
        <div style={{ display: 'flex', flexDirection: 'row', gap: 8 }}>
          <TextControl field={field} value={value} onValue={onValue} />
          <Button label="Unset" onClick={() => onValue(null)} />
        </div>
      )
      break
    case 'string_list':
    case 'string_map':
      control = (
        <CollectionControl field={field} value={value} onValue={onValue} onInvalid={onInvalid} />
      )
      break
    case 'audio_device': {
      const { selected, choices } = deviceChoices(devices, value)
      control = (
        <ChoiceControl
          field={field}
          value={selected}
          choices={choices}
          onValue={(v) =>
            onValue(
              v === DEFAULT_DEVICE ? null : v.startsWith('name:') ? v.slice(5) : Number(v.slice(6)),
            )
          }
        />
      )
      break
    }
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 14 }}>
        <div style={{ flexGrow: 1 }}>
          <Text color={C.dim} size={14}>
            {field.label}
          </Text>
        </div>
        {!equal(value, field.default ?? null) ? (
          <Button
            textOnly
            label="Reset to default"
            testId={`reset-${field.id}`}
            onClick={() => onValue(field.default ?? null)}
          />
        ) : null}
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
