// End-to-end smoke test against a real window and a real `shuvoice settings-bridge`.
// Uses an isolated config; opens a window on the current desktop.
//   SHUVOICE_BIN=../../target/debug/shuvoice bun src/e2e/smoke.ts
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { launch } from '@gpuix/react/automation'
import { isolatedEnvironment, screenshot } from './environment.ts'

const root = join(import.meta.dir, '..', '..')
const fixture = isolatedEnvironment(
  'config_version = 1\n[overlay]\nfont_size = 20\nmy_note = "keep"\n',
)
const { file } = fixture

const log = (msg: string) => console.log(`[${(performance.now() / 1000).toFixed(2)}s] ${msg}`)
const check = (cond: boolean, what: string) => {
  if (!cond) throw new Error(`check failed: ${what}`)
  log(`ok  ${what}`)
}

const app = await launch({
  command: 'bun',
  args: [join(root, 'src', 'start.tsx')],
  env: fixture.env,
})
const text = (id: string) => app.getByTestId(id).textContent()
const until = async (id: string, want: string, ms = 5000) => {
  const end = performance.now() + ms
  let got = ''
  while (performance.now() < end) {
    got = await text(id).catch(() => '')
    if (got.includes(want)) return got
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error(`${id}: wanted "${want}", got "${got}"`)
}

const gone = async (id: string, ms = 5000) => {
  const end = performance.now() + ms
  while (performance.now() < end) {
    if (
      !(await app
        .getByTestId(id)
        .element()
        .then(
          () => true,
          () => false,
        ))
    )
      return true
    await new Promise((r) => setTimeout(r, 50))
  }
  return false
}

try {
  await app.getByTestId('splash').waitFor()
  check(await gone('splash'), 'splash shows the logo, then clears once settings load')
  await app.getByTestId('sidebar-logo').waitFor()
  await app.getByTestId('apply').waitFor()
  await app.getByTestId('field-asr.asr_backend').waitFor()
  check((await text('page-title')) === 'Speech', 'opens on Speech with the engine field')
  check(
    (await text('field-asr.asr_backend')).includes('Sherpa'),
    'engine shows the effective default',
  )

  await app.getByTestId('nav-appearance').click()
  await app.getByTestId('field-overlay.font_size').waitFor()
  await app.getByTestId('field-overlay.font_size').fill('28')
  check(
    (await until('footer-status', 'unsaved')).includes('1 unsaved change'),
    'edit marks the draft dirty',
  )

  await app.getByTestId('apply').click()
  await until('footer-status', 'restart ShuVoice to apply')
  const saved = readFileSync(file, 'utf8')
  check(
    saved.includes('font_size = 28') && saved.includes('my_note = "keep"'),
    'apply patches the file, keeps unknown keys',
  )

  await app.getByTestId('field-overlay.font_size').fill('0')
  await app.getByTestId('apply').click()
  check(
    (await until('error-overlay.font_size', 'between')).length > 0,
    'out-of-range value shows a field error',
  )
  check(readFileSync(file, 'utf8') === saved, 'invalid draft is not written')

  await app.getByTestId('revert').click()
  check(!(await text('footer-status')).includes('unsaved'), 'revert discards the draft')

  await app.getByTestId('nav-service').click()
  check((await text('page-title')) === 'Service', 'service page opens')
  await app.getByTestId('nav-shortcuts').click()
  check((await text('page-title')) === 'Shortcuts', 'shortcuts page opens')
  await app.getByTestId('global-search').fill('font size')
  await app.getByTestId('result-overlay.font_size').click()
  check((await text('page-title')) === 'Appearance', 'global search jumps across sections')
  if (process.env.SHUVOICE_SETTINGS_SHOTS) {
    const dir = process.env.SHUVOICE_SETTINGS_SHOTS
    for (const page of [
      'speech',
      'vocabulary',
      'typing',
      'text_to_speech',
      'audio',
      'appearance',
      'advanced',
      'shortcuts',
      'service',
    ]) {
      await app.getByTestId(`nav-${page}`).click()
      await screenshot(page, dir)
    }
  }
  log('PASS')
} catch (error) {
  log(`FAIL ${error instanceof Error ? error.message : String(error)}`)
  process.exitCode = 1
} finally {
  await app.close()
}
