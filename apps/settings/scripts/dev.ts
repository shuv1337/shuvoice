// Run the app from source with the patched addon and hot reload.
//   SHUVOICE_BIN=../../target/debug/shuvoice bun run dev
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const root = join(import.meta.dir, '..')
const addon = join(root, 'vendor', 'gpuix-native.linux-x64-gnu.node')
if (!existsSync(addon)) {
  console.error('missing vendor/gpuix-native.linux-x64-gnu.node — run `bun run addon` first')
  process.exit(1)
}
const proc = Bun.spawn(['bun', '--hot', join(root, 'src', 'start.tsx')], {
  stdio: ['inherit', 'inherit', 'inherit'],
  env: {
    ...process.env,
    NAPI_RS_NATIVE_LIBRARY_PATH: addon,
    SHUVOICE_BIN: process.env.SHUVOICE_BIN ?? join(root, '..', '..', 'target', 'debug', 'shuvoice'),
  },
})
process.exit(await proc.exited)
