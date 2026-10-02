// Process entry: spawn the bridge and open the window.
import { createRenderer, enableAutomation, render } from '@gpuix/react'
import { App } from './app.tsx'
import { Bridge, resolveShuvoiceBin, spawnBridge } from './bridge.ts'
import { loadBrand } from './branding.ts'

const bridge = new Bridge(spawnBridge(resolveShuvoiceBin()))
const brand = await loadBrand()
const options = { title: 'ShuVoice Settings', appId: 'shuvoice-settings', width: 920, height: 640 }

if (process.env.SHUVOICE_SETTINGS_AUTOMATION === '1') {
  // Tests drive the window over stdio (see src/e2e).
  const renderer = createRenderer()
  renderer.init(options)
  enableAutomation(renderer)
  render(<App bridge={bridge} brand={brand} />, { ...options, renderer })
} else {
  render(<App bridge={bridge} brand={brand} />, options)
}
