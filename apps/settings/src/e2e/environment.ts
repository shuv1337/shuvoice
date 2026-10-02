import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'

export function isolatedEnvironment(config = 'config_version = 1\n') {
  mkdirSync('/tmp/shuvcode', { recursive: true })
  const root = mkdtempSync('/tmp/shuvcode/settings-e2e-')
  const home = join(root, 'config')
  const runtime = join(root, 'run')
  const data = join(root, 'data')
  for (const dir of [join(home, 'shuvoice'), join(home, 'hypr'), runtime, data]) {
    mkdirSync(dir, { recursive: true, mode: 0o700 })
  }
  const file = join(home, 'shuvoice/config.toml')
  const hypr = join(home, 'hypr/hyprland.conf')
  writeFileSync(file, config)
  writeFileSync(hypr, '# Settings e2e fixture; never sourced by Hyprland.\n')
  const display = process.env.WAYLAND_DISPLAY ?? 'wayland-1'
  const env = {
    ...process.env,
    // The current bridge also inspects HOME's Hyprland candidates after XDG.
    // Keep both trees private; report that backend precedence issue separately.
    HOME: root,
    SHUVOICE_BIN: resolve(process.env.SHUVOICE_BIN ?? '../../target/debug/shuvoice'),
    SHUVOICE_SETTINGS_AUTOMATION: '1',
    SHUVOICE_SETTINGS_NO_RESTART: '1',
    XDG_CONFIG_HOME: home,
    XDG_DATA_HOME: data,
    XDG_RUNTIME_DIR: runtime,
    XDG_CACHE_HOME: join(root, 'cache'),
    DBUS_SESSION_BUS_ADDRESS: `unix:path=${join(runtime, 'bus')}`,
    // The GUI alone connects to the real compositor; its bridge socket stays private.
    WAYLAND_DISPLAY: isAbsolute(display)
      ? display
      : join(process.env.XDG_RUNTIME_DIR ?? `/run/user/${process.getuid!()}`, display),
    NAPI_RS_NATIVE_LIBRARY_PATH: resolve('vendor/gpuix-native.linux-x64-gnu.node'),
  }
  return { root, file, hypr, env }
}

export async function screenshot(name: string, directory: string) {
  mkdirSync(directory, { recursive: true })
  await Bun.sleep(200)
  const sig =
    process.env.HYPRLAND_INSTANCE_SIGNATURE ??
    readFileSync('/tmp/shuvcode/pr69/live/hypr.sig', 'utf8').trim()
  const child = Bun.spawn(['hyprctl', 'clients', '-j'], {
    env: { ...process.env, HYPRLAND_INSTANCE_SIGNATURE: sig },
    stdout: 'pipe',
  })
  const clients = JSON.parse(await new Response(child.stdout).text())
  const win = clients.find((w: { class: string }) => w.class === 'shuvoice-settings')
  if (!win) throw new Error('Settings window geometry missing')
  const geometry = `${win.at[0]},${win.at[1]} ${win.size[0]}x${win.size[1]}`
  const shot = Bun.spawn(['grim', '-g', geometry, join(directory, `${name}.png`)])
  if ((await shot.exited) !== 0) throw new Error(`Screenshot failed: ${name}`)
}
