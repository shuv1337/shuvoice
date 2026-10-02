// Real bridge + real windows. All writable paths are private; no model downloads
// or shortcut confirmations are performed. Finish uses NO_RESTART=1.
import { launch } from '@gpuix/react/automation'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { Bridge, spawnBridge } from '../bridge.ts'
import type { Schema, Snapshot } from '../schema.ts'
import { isolatedEnvironment, screenshot } from './environment.ts'

const fixture = isolatedEnvironment(
  'config_version = 1\n[vocabulary]\nrecognition_hints = ["GPUIX"]\n' +
    '[typing.text_replacements]\n"g p u i" = "GPUI"\n',
)
const directory = process.env.SHUVOICE_SETTINGS_SHOTS ?? '/tmp/shuvcode/frontend-shots-v2'
const bridge = new Bridge(spawnBridge(fixture.env.SHUVOICE_BIN, fixture.env))
const check = (condition: boolean, message: string) => {
  if (!condition) throw new Error(message)
  console.log(`ok  ${message}`)
}
const schema = await bridge.call<Schema>('schema')
const snapshot = await bridge.call<Snapshot>('snapshot')
const inventory = await bridge.call<{ required: { id: string }[] }>('models', { changes: {} })
const binding = await bridge.call<{
  config_path: string | null
  error: string | null
  options: { id: string }[]
}>('shortcut_get')
check(
  binding.config_path === fixture.hypr && binding.error === null,
  'shortcut_get uses only the fixture',
)
const hyprBefore = readFileSync(fixture.hypr, 'utf8')
const dryRun = await bridge.call<{ status: string; message: string }>('shortcut_set', {
  id: binding.options[0]!.id,
  dry_run: true,
})
check(!['error', 'unsupported'].includes(dryRun.status), 'real shortcut dry-run succeeds')
check(readFileSync(fixture.hypr, 'utf8') === hyprBefore, 'shortcut dry-run does not write')
const expectedPreview = await bridge.call<{ output: string; builtins: Record<string, string> }>(
  'corrections_preview',
  { text: 'g p u i meets hyperland', changes: {} },
)
check(
  expectedPreview.output.includes('GPUI') && expectedPreview.output.includes('Hyprland'),
  'real corrections preview combines custom and built-in rules',
)
bridge.close()

const open = (onboarding = false) =>
  launch({
    command: 'bun',
    args: [join(import.meta.dir, '../start.tsx'), ...(onboarding ? ['--onboarding'] : [])],
    env: fixture.env,
  })
type Window = Awaited<ReturnType<typeof open>>
async function until(app: Window, id: string, wanted: string, timeout = 7000) {
  const end = Date.now() + timeout
  let text = ''
  while (Date.now() < end) {
    text = await app
      .getByTestId(id)
      .textContent()
      .catch(() => '')
    if (text.includes(wanted)) return text
    await Bun.sleep(80)
  }
  throw new Error(`${id}: expected ${JSON.stringify(wanted)}, got ${JSON.stringify(text)}`)
}
async function ready(app: Window) {
  await app.getByTestId('apply').waitFor()
  const end = Date.now() + 7000
  while (
    await app
      .getByTestId('splash')
      .element()
      .then(
        () => true,
        () => false,
      )
  ) {
    if (Date.now() > end) throw new Error('Splash did not clear')
    await Bun.sleep(100)
  }
}

const app = await open()
try {
  await ready(app)
  const kinds = new Set<string>()
  // Search also reaches fields hidden by another engine/provider or an advanced group.
  for (const field of schema.fields) {
    await app.getByTestId('global-search').fill(field.id)
    await app.getByTestId(`result-${field.id}`).click()
    const collection = ['string_list', 'string_map'].includes(field.kind.type)
    await app.getByTestId(`${collection ? 'search' : 'field'}-${field.id}`).waitFor()
    kinds.add(field.kind.type)
  }
  check(
    kinds.size === 9,
    `${schema.fields.length} real schema fields render; all nine kinds reached`,
  )

  await app.getByTestId('global-search').fill('overlay.font_size')
  await app.getByTestId('result-overlay.font_size').click()
  check(
    !(await app
      .getByTestId('reset-overlay.font_size')
      .element()
      .then(
        () => true,
        () => false,
      )),
    'default field has no reset link',
  )
  await app.getByTestId('field-overlay.font_size').fill('28')
  await app.getByTestId('reset-overlay.font_size').waitFor()
  await screenshot('reset-link', directory)
  await app.getByTestId('reset-overlay.font_size').click()
  await Bun.sleep(100)
  check(
    !(await app
      .getByTestId('reset-overlay.font_size')
      .element()
      .then(
        () => true,
        () => false,
      )),
    'reset restores the default and hides the link',
  )

  for (const page of [...schema.sections, 'shortcuts', 'service']) {
    await app.getByTestId(`nav-${page}`).click()
    if (page === 'advanced') await app.getByTestId('advanced-toggle').click()
    if (page === 'speech') {
      for (const model of inventory.required) await app.getByTestId(`model-${model.id}`).waitFor()
      check(inventory.required.length > 0, 'real model inventory renders')
    }
    if (page === 'vocabulary') {
      await until(app, 'hint-support', 'Hints unsupported')
      await app.getByTestId('correction-preview').fill('g p u i meets hyperland')
      await until(app, 'correction-output', expectedPreview.output)
      await app.getByTestId('builtins-toggle').click()
      await until(app, 'builtins', 'hyperland')
      await app.getByTestId('builtins-toggle').click()
      check(true, 'real capability and corrections preview render')
    }
    if (page === 'shortcuts') {
      await app.getByTestId('shortcut-preview').click()
      await until(app, 'shortcut-result', dryRun.message)
      await app.getByTestId('shortcut-confirm').waitFor()
      check(
        readFileSync(fixture.hypr, 'utf8') === hyprBefore,
        'UI preview leaves fixture unchanged',
      )
    }
    await screenshot(page, directory)
  }
  // Capture the real map editor without 31 built-in rows hiding the custom rule.
  await app.getByTestId('global-search').fill('typing.text_replacements')
  await app.getByTestId('result-typing.text_replacements').click()
  await app.getByTestId('search-typing.text_replacements').fill('g p u i')
  await screenshot('corrections-editor', directory)

  // Changing engines must recompute capability against the draft, without saving.
  await app.getByTestId('nav-speech').click()
  await app.getByTestId('field-asr.asr_backend').click()
  await app.getByTestId('option-asr.asr_backend-openai_realtime').click()
  await app.getByTestId('nav-vocabulary').click()
  await until(app, 'hint-support', 'Hints supported')
  check(true, 'capability follows the unsaved engine draft')
  await app.getByTestId('add-vocabulary.recognition_hints').click()
  await Bun.sleep(450)
  await app.getByTestId('entry-vocabulary.recognition_hints-1-0').fill('Hyprland')
  await Bun.sleep(650)
  check(
    !(await app
      .getByTestId('operation-error')
      .element()
      .then(
        () => true,
        () => false,
      )),
    'fixing an incomplete vocabulary entry clears draft operation errors',
  )
} finally {
  await app.close()
}

const onboarding = await open(true)
try {
  await ready(onboarding)
  for (let step = 0; step < 5; step++) {
    await screenshot(`onboarding-${step}`, directory)
    if (step < 4) await onboarding.getByTestId('onboarding-next').click()
  }
  await onboarding.getByTestId('apply').click()
  await until(onboarding, 'footer-status', 'restart ShuVoice to apply')
  check(
    existsSync(join(fixture.env.XDG_DATA_HOME, 'shuvoice/.wizard-done')),
    'onboarding Finish writes the isolated marker without restarting',
  )
  const verify = new Bridge(spawnBridge(fixture.env.SHUVOICE_BIN, fixture.env))
  try {
    const saved = await verify.call<Snapshot>('snapshot')
    check(
      saved.values['asr.sherpa_profile'] === 'instant',
      'onboarding saves the wizard engine default',
    )
    check(
      saved.values['tts.tts_backend'] === 'kokoro',
      'onboarding saves the wizard read-aloud default',
    )
    check(
      JSON.stringify(saved.values['vocabulary.recognition_hints']) ===
        JSON.stringify(snapshot.values['vocabulary.recognition_hints']),
      'onboarding preserves explicit vocabulary',
    )
  } finally {
    verify.close()
  }
  await screenshot('onboarding-result', directory)
  console.log('PASS: real-bridge features and onboarding')
} finally {
  await onboarding.close()
}
