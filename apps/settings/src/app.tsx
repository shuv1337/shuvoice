import { SettingsSection } from './screens/settings-section.tsx'
import { ApplyFooter } from './components/apply-footer.tsx'
import { Models } from './screens/models.tsx'
import { Shortcuts } from './screens/shortcuts.tsx'
import { Vocabulary } from './screens/vocabulary.tsx'
import { useSettings } from './use-settings.ts'
import { AnimatePresence } from '@gpuix/react'
import type { Bridge } from './bridge.ts'
import type { Brand } from './branding.ts'
import { searchFields } from './draft.ts'
import { SECTION_LABELS, type Section } from './schema.ts'
import { C, INPUT_STYLE } from './theme.ts'
import { Text, Button } from './components/controls.tsx'
import { Splash } from './components/splash.tsx'
import { ServicePage } from './screens/service.tsx'
import type { Page } from './apply-state.ts'
import { OnboardingNavigation, ONBOARDING_STAGES } from './screens/onboarding.tsx'

export function App({
  bridge,
  brand,
  onboarding: initialOnboarding = false,
}: {
  bridge: Bridge
  brand: Brand
  onboarding?: boolean
}) {
  const settings = useSettings(bridge, initialOnboarding)
  const {
    onboarding,
    setOnboarding,
    stage,
    setStage,
    query,
    setQuery,
    highlight,
    setHighlight,
    setAdvanced,
    operationError,
    feature,
    schema,
    snap,
    errors,
    status,
    page,
    setPage,
    phase,
    splashHeld,
  } = settings

  const pages: Page[] = [
    ...new Set<Section>([...(schema?.sections ?? []), 'vocabulary', 'advanced']),
    'shortcuts',
    'service',
  ]
  const showSplash = splashHeld || phase.kind === 'loading'
  const moveStage = (next: number) => {
    setStage(next)
    setPage(ONBOARDING_STAGES[next]!.page)
    setHighlight('')
  }

  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'row',
        width: '100%',
        height: '100%',
        backgroundColor: C.bg,
      }}
    >
      <div
        style={{
          width: 200,
          flexShrink: 0,
          backgroundColor: C.side,
          padding: 12,
          gap: 4,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          testId="sidebar-logo"
          style={{ paddingLeft: 4, paddingRight: 4, paddingTop: 4, paddingBottom: 12 }}
        >
          <img
            src={brand.lockup}
            alt="ShuVoice"
            objectFit="contain"
            style={{ width: 168, height: 73 }}
          />
        </div>
        {onboarding ? (
          <OnboardingNavigation
            stage={stage}
            moveStage={moveStage}
            skip={() => setOnboarding(false)}
          />
        ) : (
          pages.map((p) => {
            const label =
              p === 'service' ? 'Service' : p === 'shortcuts' ? 'Shortcuts' : SECTION_LABELS[p]
            const pageHasError =
              p !== 'service' && schema?.fields.some((f) => f.section === p && errors[f.id])
            return (
              <div
                key={p}
                testId={`nav-${p}`}
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'enter' || e.key === 'space') setPage(p)
                }}
                onClick={() => {
                  setPage(p)
                  setHighlight('')
                  setAdvanced(false)
                }}
                style={{
                  padding: 8,
                  borderRadius: 6,
                  cursor: 'pointer',
                  backgroundColor: p === page ? C.card : C.side,
                  hover: { backgroundColor: C.card },
                }}
              >
                <Text color={pageHasError ? C.bad : p === page ? C.text : C.dim}>{label}</Text>
              </div>
            )
          })
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minWidth: 0 }}>
        {!onboarding ? (
          <input
            testId="global-search"
            placeholder="Search settings"
            value={query}
            onChange={(e) => setQuery(e.value ?? '')}
            style={{ ...INPUT_STYLE, margin: 16 }}
          />
        ) : null}
        <div
          key={`${page}-${highlight}`}
          style={{
            flexGrow: 1,
            minHeight: 0,
            overflowY: 'scroll',
            padding: 24,
            gap: 20,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {searchFields(schema?.fields ?? [], query).map((f) => (
            <Button
              key={f.id}
              label={`${SECTION_LABELS[f.section]} · ${f.label}`}
              testId={`result-${f.id}`}
              onClick={() => {
                setPage(f.section)
                setHighlight(f.id)
                setAdvanced(true)
                setQuery('')
              }}
            />
          ))}
          <Text bold size={20} testId="page-title">
            {onboarding
              ? ONBOARDING_STAGES[stage]?.title
              : page === 'service'
                ? 'Service'
                : page === 'shortcuts'
                  ? 'Shortcuts'
                  : SECTION_LABELS[page]}
          </Text>
          {operationError ? (
            <Text color={C.bad} testId="operation-error">
              {operationError}
            </Text>
          ) : null}

          {onboarding && !feature('onboarding_defaults') ? (
            <Text color={C.warn}>
              Setup requires a newer ShuVoice bridge. Open settings or run shuvoice wizard.
            </Text>
          ) : null}
          <Models {...settings} />
          {page === 'vocabulary' ? <Vocabulary {...settings} /> : null}
          {page === 'shortcuts' ? <Shortcuts {...settings} /> : null}
          {snap?.config_error ? (
            <div style={{ padding: 10, borderRadius: 6, backgroundColor: '#3a2228' }}>
              <Text
                color={C.bad}
                testId="config-error"
              >{`Current config is invalid: ${snap.config_error}`}</Text>
            </div>
          ) : null}
          {page === 'service' ? (
            <ServicePage status={status} snap={snap} />
          ) : (
            <SettingsSection {...settings} />
          )}
        </div>

        <ApplyFooter {...settings} moveStage={moveStage} />
      </div>
      <AnimatePresence>
        {showSplash ? <Splash key="splash" src={brand.splash} /> : null}
      </AnimatePresence>
    </div>
  )
}
