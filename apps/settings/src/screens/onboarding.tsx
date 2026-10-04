import { Text, Button } from '../components/button.tsx'

export const ONBOARDING_STAGES = [
  { title: 'Engine', page: 'speech' },
  { title: 'Microphone', page: 'audio' },
  { title: 'Read-aloud', page: 'text_to_speech' },
  { title: 'Shortcut', page: 'shortcuts' },
  { title: 'Finish', page: 'service' },
] as const

export function OnboardingNavigation({
  stage,
  moveStage,
  skip,
}: {
  stage: number
  moveStage: (next: number) => void
  skip: () => void
}) {
  return (
    <>
      <Text bold>Set up ShuVoice</Text>
      {ONBOARDING_STAGES.map(({ title }, i) => (
        <Button key={title} label={title} primary={i === stage} onClick={() => moveStage(i)} />
      ))}
      <Button label="Skip to settings" testId="skip-onboarding" onClick={skip} />
    </>
  )
}
