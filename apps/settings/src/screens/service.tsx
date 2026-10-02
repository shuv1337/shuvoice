import { C } from '../theme.ts'
import { Text } from '../components/controls.tsx'
import type { ServiceStatus, Snapshot } from '../schema.ts'

export function ServicePage({
  status,
  snap,
}: {
  status: ServiceStatus | null
  snap: Snapshot | null
}) {
  const rows: [string, string][] = [
    ['Service', status ? status.active_state : '…'],
    [
      'Overlay',
      status?.ui_ready === true
        ? 'Ready'
        : status?.active_state === 'active' && status.ui_ready === false
          ? 'Starting'
          : '—',
    ],
    ['Dictation', status?.stt ?? '—'],
    ['Read aloud', status?.tts ?? '—'],
  ]
  if (snap) rows.push(['Config file', snap.path])
  if (snap?.config_error) rows.push(['Config error', snap.config_error])
  for (const secret of snap?.secrets ?? [])
    rows.push([
      secret.env,
      secret.present ? `Set · ${secret.source ?? 'source unavailable'}` : 'Not set',
    ])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: 'flex', flexDirection: 'row', gap: 12 }}>
          <div style={{ width: 180, flexShrink: 0 }}>
            <Text color={C.dim}>{label}</Text>
          </div>
          <Text testId={`status-${label}`}>{value}</Text>
        </div>
      ))}
    </div>
  )
}

export function serviceSummary(status: ServiceStatus | null): { text: string; color: string } {
  if (!status) return { text: 'Service …', color: C.dim }
  if (status.active_state === 'active') {
    return status.ui_ready === false
      ? { text: 'Service starting', color: C.warn }
      : { text: 'Service running', color: C.ok }
  }
  if (status.active_state === 'failed') return { text: 'Service failed', color: C.bad }
  return { text: `Service ${status.active_state}`, color: C.dim }
}
