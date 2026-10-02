import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@gpuix/react/select'
import { C } from '../theme.ts'
import { Text } from './button.tsx'
import type { Choice, FieldMeta, Json } from '../schema.ts'

const CHEVRON = (color: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`
const CHECK = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="${C.accent}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`

const CONTROL_WIDTH = 320

export function ChoiceControl({
  field,
  value,
  choices,
  onValue,
}: {
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
          boxShadow: {
            offsetX: 0,
            offsetY: 8,
            blurRadius: 24,
            spreadRadius: 0,
            color: '#00000099',
          },
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
