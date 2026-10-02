import { expect, test } from 'bun:test'
import { readFileSync, rmSync, statSync } from 'node:fs'
import { isAbsolute } from 'node:path'
import { isolatedEnvironment } from './environment.ts'

test('real-window fixtures isolate bridge state while preserving an absolute compositor socket', () => {
  const fixture = isolatedEnvironment()
  try {
    for (const name of [
      'HOME',
      'XDG_CONFIG_HOME',
      'XDG_DATA_HOME',
      'XDG_RUNTIME_DIR',
      'XDG_CACHE_HOME',
    ] as const) {
      expect(fixture.env[name].startsWith(fixture.root)).toBe(true)
      expect(fixture.env[name]).not.toBe(process.env[name])
    }
    expect(fixture.env.SHUVOICE_SETTINGS_NO_RESTART).toBe('1')
    expect(fixture.env.DBUS_SESSION_BUS_ADDRESS).toContain(fixture.env.XDG_RUNTIME_DIR)
    expect(isAbsolute(fixture.env.WAYLAND_DISPLAY)).toBe(true)
    expect(statSync(fixture.env.XDG_RUNTIME_DIR).mode & 0o777).toBe(0o700)
    expect(readFileSync(fixture.hypr, 'utf8')).toContain('never sourced by Hyprland')
  } finally {
    rmSync(fixture.root, { recursive: true })
  }
})
