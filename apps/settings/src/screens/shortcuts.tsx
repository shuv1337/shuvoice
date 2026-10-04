import { C } from '../theme.ts'
import { Text, Button, ChoiceControl } from '../components/controls.tsx'
import type { SettingsState } from '../use-settings.ts'

export function Shortcuts({
  feature,
  shortcut,
  binding,
  setBinding,
  setShortcutPreview,
  setShortcutResult,
  shortcutBusy,
  setShortcutBinding,
  shortcutResult,
  shortcutPreview,
}: SettingsState) {
  return (
    <>
      <>
        {!feature('shortcut_get') || !feature('shortcut_set') ? (
          <Text color={C.dim}>Shortcut editing requires a newer ShuVoice bridge.</Text>
        ) : (
          <>
            <Text>{`Push-to-talk: ${shortcut?.current?.label ?? 'Not configured'}`}</Text>
            {shortcut?.error ? <Text color={C.warn}>{shortcut.error}</Text> : null}
            {shortcut?.config_path ? <Text color={C.dim}>{shortcut.config_path}</Text> : null}
            <ChoiceControl
              field={{
                id: 'shortcut',
                section: 'advanced',
                label: 'Shortcut',
                help: '',
                unit: '',
                kind: { type: 'choice', choices: [] },
              }}
              value={binding}
              choices={(shortcut?.options ?? []).map((o) => ({ value: o.id, label: o.label }))}
              onValue={(v) => {
                setBinding(v)
                setShortcutPreview(null)
                setShortcutResult(null)
              }}
            />
            <Text color={C.dim}>Applies immediately to Hyprland.</Text>
            <Button
              label="Preview shortcut"
              testId="shortcut-preview"
              disabled={!binding || !!shortcut?.error || shortcutBusy}
              onClick={() => void setShortcutBinding(true)}
            />
            {shortcutResult ? (
              <>
                <Text testId="shortcut-result">{shortcutResult.message}</Text>
                {shortcutResult.conflicts.map((c) => (
                  <Text key={c} color={C.warn}>
                    {c}
                  </Text>
                ))}
                {shortcutResult.backup ? <Text>{`Backup: ${shortcutResult.backup}`}</Text> : null}
              </>
            ) : null}
            {shortcutPreview === binding ? (
              <>
                <Button
                  label="Confirm shortcut"
                  testId="shortcut-confirm"
                  disabled={shortcutBusy}
                  onClick={() => void setShortcutBinding(false)}
                />
                <Button label="Cancel" onClick={() => setShortcutPreview(null)} />
              </>
            ) : null}
          </>
        )}
      </>
    </>
  )
}
