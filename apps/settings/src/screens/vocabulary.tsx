import { useState } from 'react'
import { C, INPUT_STYLE } from '../theme.ts'
import { Text, Button } from '../components/controls.tsx'
import type { SettingsState } from '../use-settings.ts'

export function Vocabulary({
  hints,
  feature,
  previewText,
  setPreviewText,
  preview,
}: SettingsState) {
  const [showBuiltins, setShowBuiltins] = useState(false)
  const builtins = Object.entries(preview?.builtins ?? {})
  return (
    <>
      <>
        {hints ? (
          <>
            <Text testId="hint-support" color={hints.supported ? C.ok : C.warn}>
              {hints.supported ? 'Hints supported' : 'Hints unsupported'}
            </Text>
            <Text color={C.dim}>{hints.detail}</Text>
          </>
        ) : (
          <Text color={C.dim}>Hint support unavailable from this bridge.</Text>
        )}
        {feature('corrections_preview') ? (
          <>
            <Text bold>Correction preview</Text>
            <input
              testId="correction-preview"
              value={previewText}
              placeholder="Try a transcription"
              onChange={(e) => setPreviewText(e.value ?? '')}
              style={INPUT_STYLE}
            />
            <Text testId="correction-output">{preview?.output ?? ''}</Text>
            <div style={{ display: 'flex', flexDirection: 'row' }}>
              <Button
                textOnly
                testId="builtins-toggle"
                label={`${showBuiltins ? 'Hide' : 'Show'} built-in corrections (${builtins.length})`}
                onClick={() => setShowBuiltins(!showBuiltins)}
              />
            </div>
            {showBuiltins ? (
              <div
                testId="builtins"
                style={{ display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0 }}
              >
                {builtins.map(([key, value]) => (
                  <Text key={key} color={C.dim}>{`${key} → ${value}`}</Text>
                ))}
              </div>
            ) : null}
          </>
        ) : null}
      </>
    </>
  )
}
