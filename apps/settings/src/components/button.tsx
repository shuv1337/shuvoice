import type { ReactNode } from 'react'
import { C } from '../theme.ts'

export function Text({
  children,
  color = C.text,
  size,
  bold,
  testId,
}: {
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

export function Button({
  label,
  onClick,
  primary,
  disabled,
  testId,
  textOnly,
}: {
  label: string
  onClick: () => void
  primary?: boolean
  disabled?: boolean
  testId?: string
  textOnly?: boolean
}) {
  const bg = primary ? C.accent : C.card
  return (
    <div
      testId={testId}
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'enter' || e.key === 'space')) onClick()
      }}
      onClick={disabled ? undefined : onClick}
      style={{
        paddingLeft: textOnly ? 0 : 14,
        paddingRight: textOnly ? 0 : 14,
        paddingTop: textOnly ? 2 : 8,
        paddingBottom: textOnly ? 2 : 8,
        flexShrink: 0,
        borderRadius: 6,
        cursor: disabled ? 'default' : 'pointer',
        opacity: disabled ? 0.45 : 1,
        backgroundColor: textOnly ? 'transparent' : bg,
        hover:
          disabled || textOnly ? undefined : { backgroundColor: primary ? '#8fb2fa' : C.cardHover },
      }}
    >
      <Text
        color={textOnly ? C.accent : primary ? C.onAccent : C.text}
        size={textOnly ? 13 : undefined}
        bold={primary}
      >
        {label}
      </Text>
    </div>
  )
}
