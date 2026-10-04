import { C } from '../theme.ts'
import { Text, Button } from '../components/controls.tsx'
import type { SettingsState } from '../use-settings.ts'

export function Models({
  page,
  feature,
  models,
  busy,
  downloadModel,
  download,
  cancel,
}: SettingsState) {
  return (
    <>
      {page === 'speech' && feature('models') ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {models.map((m) => (
            <div
              key={m.id}
              style={{ display: 'flex', flexDirection: 'row', gap: 12, alignItems: 'center' }}
            >
              <Text
                testId={`model-${m.id}`}
              >{`${m.label} · ${m.installed ? 'Installed' : (m.size_hint ?? 'Not installed')}`}</Text>
              {!m.installed && feature('model_download') ? (
                <Button
                  label="Download"
                  testId={`download-${m.id}`}
                  disabled={busy}
                  onClick={() => void downloadModel(m.id)}
                />
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {download ? (
        <>
          <Text>{download.text}</Text>
          <div style={{ height: 6, backgroundColor: C.line }}>
            <div
              style={{
                height: 6,
                width: `${Math.max(0, Math.min(1, download.fraction ?? 0.1)) * 100}%`,
                backgroundColor: C.accent,
              }}
            />
          </div>
          <Button label="Cancel download" onClick={cancel} />
        </>
      ) : null}
    </>
  )
}
