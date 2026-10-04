import { useEffect, useRef, useState } from 'react'
import { INPUT_STYLE } from '../theme.ts'
import { Button } from './button.tsx'
import type { FieldMeta, Json } from '../schema.ts'

export function CollectionControl({
  field,
  value,
  onValue,
  onInvalid,
}: {
  field: FieldMeta
  value: Json
  onValue: (v: Json) => void
  onInvalid: (m: string) => void
}) {
  const map = field.kind.type === 'string_map'
  const decode = (v: Json): string[][] =>
    map
      ? Object.entries(v && !Array.isArray(v) && typeof v === 'object' ? v : {}).map(([k, v]) => [
          k,
          String(v),
        ])
      : (Array.isArray(v) ? v : []).map((v) => [String(v)])
  const [rows, setRows] = useState(() => decode(value))
  const [query, setQuery] = useState('')
  const sent = useRef(value)
  useEffect(() => {
    if (value !== sent.current) {
      setRows(decode(value))
      sent.current = value
    }
  }, [value])
  const update = (next: string[][]) => {
    setRows(next)
    // JSON objects cannot represent duplicate keys. Keep these rows locally
    // until they can be sent losslessly; Rust validates all representable values.
    if (map && new Set(next.map((r) => r[0])).size !== next.length) {
      onInvalid('Duplicate correction')
      return
    }
    const nextValue = map ? Object.fromEntries(next) : next.map((r) => r[0] ?? '')
    sent.current = nextValue
    onValue(nextValue)
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <input
        testId={`search-${field.id}`}
        placeholder="Filter entries"
        value={query}
        onChange={(e) => setQuery(e.value ?? '')}
        style={{ ...INPUT_STYLE, width: 320 }}
      />
      {rows.map((row, i) =>
        row.join(' ').toLowerCase().includes(query.toLowerCase()) ? (
          <div key={i} style={{ display: 'flex', flexDirection: 'row', gap: 8 }}>
            {row.map((text, j) => (
              <input
                key={j}
                testId={`entry-${field.id}-${i}-${j}`}
                value={text}
                onChange={(e) =>
                  update(
                    rows.map((r, n) =>
                      n === i ? r.map((v, k) => (k === j ? (e.value ?? '') : v)) : r,
                    ),
                  )
                }
                style={{ ...INPUT_STYLE, width: map ? 220 : 360 }}
              />
            ))}
            <Button label="Remove" onClick={() => update(rows.filter((_, n) => n !== i))} />
          </div>
        ) : null,
      )}
      <Button
        label={map ? 'Add correction' : 'Add term'}
        testId={`add-${field.id}`}
        onClick={() => {
          setQuery('')
          update([...rows, map ? ['', ''] : ['']])
        }}
      />
    </div>
  )
}
