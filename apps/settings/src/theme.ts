export const C = {
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

export const INPUT_STYLE = {
  padding: 8,
  minHeight: 36,
  flexShrink: 0,
  borderRadius: 6,
  borderWidth: 1,
  borderColor: C.line,
  backgroundColor: C.field,
  color: C.text,
} as const
