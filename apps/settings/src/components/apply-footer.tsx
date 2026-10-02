import { C } from '../theme.ts'
import { Text, Button } from './controls.tsx'
import { progressText } from '../apply-state.ts'
import { serviceSummary } from '../screens/service.tsx'
import type { SettingsState } from '../use-settings.ts'

export function ApplyFooter({
  status,
  phase,
  errors,
  dirty,
  load,
  cancel,
  models,
  onboarding,
  stage,
  busy,
  hasErrors,
  revert,
  feature,
  apply,
  moveStage,
}: SettingsState & { moveStage: (next: number) => void }) {
  const summary = serviceSummary(status)

  let footerText = ''
  let footerColor = C.dim
  if (phase.kind === 'loading') footerText = 'Loading…'
  else if (phase.kind === 'applying') {
    footerText = progressText(phase)
    footerColor = phase.step === 'waiting_idle' ? C.warn : C.dim
  } else if (phase.kind === 'done')
    [footerText, footerColor] = [phase.text, phase.tone === 'ok' ? C.ok : C.warn]
  else if (phase.kind === 'conflict') [footerText, footerColor] = ['Config changed on disk', C.warn]
  else if (phase.kind === 'error') [footerText, footerColor] = [phase.message, C.bad]
  else if (errors['']) [footerText, footerColor] = [errors[''], C.bad]
  else if (dirty > 0)
    [footerText, footerColor] = [
      dirty === 1 ? '1 unsaved change' : `${dirty} unsaved changes`,
      C.warn,
    ]

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
        padding: 14,
        borderTopWidth: 1,
        borderColor: C.line,
        backgroundColor: C.side,
      }}
    >
      <div
        style={{
          flexGrow: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'row',
          gap: 12,
          alignItems: 'center',
        }}
      >
        <Text color={footerColor} testId="footer-status">
          {footerText}
        </Text>
        {phase.kind === 'conflict' ? (
          <Button label="Reload" onClick={() => void load()} testId="reload" />
        ) : null}
        {phase.kind === 'applying' && phase.step === 'waiting_idle' ? (
          <Button label="Cancel" onClick={cancel} testId="cancel" />
        ) : null}
      </div>
      <Text color={summary.color} testId="service-summary">
        {summary.text}
      </Text>
      {models.some((m) => !m.installed) ? <Text color={C.warn}>Required model missing</Text> : null}
      {onboarding && stage > 0 ? (
        <Button label="Back" disabled={busy} onClick={() => moveStage(stage - 1)} />
      ) : null}
      {onboarding && stage < 4 ? (
        <Button
          label="Next"
          testId="onboarding-next"
          primary
          disabled={busy}
          onClick={() => moveStage(stage + 1)}
        />
      ) : null}
      {(dirty > 0 || hasErrors) && !busy ? (
        <Button label="Revert" onClick={revert} testId="revert" />
      ) : null}
      <Button
        label={onboarding ? 'Finish setup' : 'Apply & Restart'}
        primary
        disabled={
          (!onboarding && dirty === 0) ||
          hasErrors ||
          busy ||
          (onboarding && (stage !== 4 || !feature('onboarding_defaults')))
        }
        onClick={() => void apply()}
        testId="apply"
      />
    </div>
  )
}
