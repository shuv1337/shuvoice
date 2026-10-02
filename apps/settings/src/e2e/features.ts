import { launch } from '@gpuix/react/automation'
import { mkdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = join(import.meta.dir, '../..')
const dir = process.env.SHUVOICE_SETTINGS_SHOTS ?? '/tmp/shuvcode/frontend-shots'
mkdirSync(dir, { recursive: true })
const app = await launch({ command: 'bun', args: [join(import.meta.dir, 'fixture-window.tsx'), '--onboarding'], env: { NAPI_RS_NATIVE_LIBRARY_PATH: join(root, 'vendor/gpuix-native.linux-x64-gnu.node') } })
const shot = async (name: string) => {
  await Bun.sleep(250)
  const clients = JSON.parse(await new Response(Bun.spawn(['hyprctl', 'clients', '-j'], { env: { ...process.env, HYPRLAND_INSTANCE_SIGNATURE: process.env.HYPRLAND_INSTANCE_SIGNATURE ?? readFileSync('/tmp/shuvcode/pr69/live/hypr.sig', 'utf8').trim() }, stdout: 'pipe' }).stdout).text())
  const w = clients.find((w: { class: string }) => w.class === 'shuvoice-settings')
  if (!w) throw new Error('Window not found')
  if (await Bun.spawn(['grim', '-g', `${w.at[0]},${w.at[1]} ${w.size[0]}x${w.size[1]}`, join(dir, `${name}.png`)]).exited) throw new Error('Screenshot failed')
}
try {
  await app.getByTestId('onboarding-next').waitFor()
  await Bun.sleep(1400)
  await app.getByTestId('download-parakeet').click()
  const deadline = Date.now() + 5000
  while (!(await app.getByTestId('model-parakeet').textContent()).includes('Installed')) {
    if (Date.now() > deadline) throw new Error('Model inventory did not refresh after download')
    await Bun.sleep(100)
  }
  for (let step = 0; step < 5; step++) {
    await shot(`onboarding-${step}`)
    if (step < 4) await app.getByTestId('onboarding-next').click()
  }
  await app.getByTestId('apply').click()
  await app.getByTestId('nav-vocabulary').waitFor()
  for (const page of ['speech', 'vocabulary', 'audio', 'text_to_speech', 'typing', 'appearance', 'advanced', 'shortcuts', 'service']) {
    await app.getByTestId(`nav-${page}`).click()
    if (page === 'advanced') await app.getByTestId('advanced-toggle').click()
    if (page === 'vocabulary') {
      await app.getByTestId('correction-preview').fill('shoe voice')
      await app.getByTestId('add-vocabulary.terms').click()
      await app.getByTestId('entry-vocabulary.terms-2-0').fill('GPUIX')
    }
    if (page === 'shortcuts') {
      await app.getByTestId('shortcut-preview').click()
      await app.getByTestId('shortcut-confirm').waitFor()
      await app.getByTestId('shortcut-confirm').click()
    }
    await shot(`fixture-${page}`)
    if (page === 'vocabulary') {
      await app.getByTestId('global-search').fill('typing.text_replacements')
      await app.getByTestId('result-typing.text_replacements').click()
      await app.getByTestId('entry-typing.text_replacements-0-1').fill('ShuVoice Settings')
      await shot('fixture-corrections')
    }
  }
  console.log('PASS: contract fixture onboarding, collections, preview, shortcut, and all pages')
} finally { await app.close() }
