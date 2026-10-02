// End-to-end smoke test against a real window and a real `shuvoice settings-bridge`.
// Uses an isolated config; opens a window on the current desktop.
//   SHUVOICE_BIN=../../target/debug/shuvoice bun src/e2e/smoke.ts
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { launch } from '@gpuix/react/automation'

const root = join(import.meta.dir, '..', '..')
const config = mkdtempSync('/tmp/shuvcode/shuvoice-settings-e2e-')
const file = join(config, 'shuvoice', 'config.toml')
mkdirSync(join(config, 'shuvoice'))
writeFileSync(file, 'config_version = 1\n[overlay]\nfont_size = 20\nmy_note = "keep"\n')

const log = (msg: string) => console.log(`[${(performance.now() / 1000).toFixed(2)}s] ${msg}`)
const check = (cond: boolean, what: string) => {
  if (!cond) throw new Error(`check failed: ${what}`)
  log(`ok  ${what}`)
}

const app = await launch({
  command: 'bun',
  args: [join(root, 'src', 'start.tsx')],
  env: {
    SHUVOICE_SETTINGS_AUTOMATION: '1',
    SHUVOICE_SETTINGS_NO_RESTART: '1',
    XDG_CONFIG_HOME: config,
    XDG_DATA_HOME: join(config, 'data'),
    NAPI_RS_NATIVE_LIBRARY_PATH: join(root, 'vendor', 'gpuix-native.linux-x64-gnu.node'),
    SHUVOICE_BIN: process.env.SHUVOICE_BIN ?? join(root, '..', '..', 'target', 'debug', 'shuvoice'),
  },
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
    if (!(await app.getByTestId(id).element().then(() => true, () => false))) return true
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
  check((await text('field-asr.asr_backend')).includes('Sherpa'), 'engine shows the effective default')

  await app.getByTestId('nav-appearance').click()
  await app.getByTestId('field-overlay.font_size').waitFor()
  await app.getByTestId('field-overlay.font_size').fill('28')
  check((await until('footer-status', 'unsaved')).includes('1 unsaved change'), 'edit marks the draft dirty')

  await app.getByTestId('apply').click()
  await until('footer-status', 'restart ShuVoice to apply')
  const saved = readFileSync(file, 'utf8')
  check(saved.includes('font_size = 28') && saved.includes('my_note = "keep"'), 'apply patches the file, keeps unknown keys')

  await app.getByTestId('field-overlay.font_size').fill('0')
  await app.getByTestId('apply').click()
  check((await until('error-overlay.font_size', 'between')).length > 0, 'out-of-range value shows a field error')
  check(readFileSync(file, 'utf8') === saved, 'invalid draft is not written')

  await app.getByTestId('revert').click()
  check(!(await text('footer-status')).includes('unsaved'), 'revert discards the draft')

  await app.getByTestId('nav-service').click()
  check((await text('page-title')) === 'Service', 'service page opens')
  await app.getByTestId('nav-shortcuts').click()
  check((await text('page-title')) === 'Shortcuts', 'old bridge still opens degraded shortcuts page')
  await app.getByTestId('global-search').fill('font size')
  await app.getByTestId('result-overlay.font_size').click()
  check((await text('page-title')) === 'Appearance', 'global search jumps across sections')
  if (process.env.SHUVOICE_SETTINGS_SHOTS) {
    const dir = process.env.SHUVOICE_SETTINGS_SHOTS
    mkdirSync(dir, { recursive: true })
    const sig = process.env.HYPRLAND_INSTANCE_SIGNATURE ?? readFileSync('/tmp/shuvcode/pr69/live/hypr.sig', 'utf8').trim()
    for (const page of ['speech', 'vocabulary', 'typing', 'text_to_speech', 'audio', 'appearance', 'advanced', 'shortcuts', 'service']) {
      await app.getByTestId(`nav-${page}`).click()
      await Bun.sleep(200)
      const clients = JSON.parse(await new Response(Bun.spawn(['hyprctl', 'clients', '-j'], { env: { ...process.env, HYPRLAND_INSTANCE_SIGNATURE: sig }, stdout: 'pipe' }).stdout).text())
      const win = clients.find((w: { class: string }) => w.class === 'shuvoice-settings')
      if (!win) throw new Error('settings window geometry missing')
      const shot = Bun.spawn(['grim', '-g', `${win.at[0]},${win.at[1]} ${win.size[0]}x${win.size[1]}`, join(dir, `${page}.png`)])
      check(await shot.exited === 0, `screenshot ${page}`)
    }
  }
  log('PASS')
} catch (error) {
  log(`FAIL ${error instanceof Error ? error.message : String(error)}`)
  process.exitCode = 1
} finally {
  await app.close()
}
