// Process entry: spawn the bridge and open the window.
import { createRenderer, enableAutomation, render } from '@gpuix/react'
import { App } from './app.tsx'
import { Bridge, resolveShuvoiceBin, spawnBridge } from './bridge.ts'

const bridge = new Bridge(spawnBridge(resolveShuvoiceBin()))
const options = { title: 'ShuVoice Settings', width: 920, height: 640 }

if (process.env.SHUVOICE_SETTINGS_AUTOMATION === '1') {
  // Tests drive the window over stdio (see src/e2e).
  const renderer = createRenderer()
  renderer.init(options)
  enableAutomation(renderer)
  render(<App bridge={bridge} />, { ...options, renderer })
} else {
  render(<App bridge={bridge} />, options)
}
