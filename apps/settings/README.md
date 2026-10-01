# ShuVoice Settings

Settings window for ShuVoice (GPUIX + React). Rust owns the settings: the app
spawns `shuvoice settings-bridge` and edits only through it.

## Build

```bash
bun install
bun run addon      # patched GPUIX native addon -> vendor/ (~75 s, needs Rust)
bun run build      # -> dist/shuvoice-settings (single file, addon embedded)
```

Install next to the `shuvoice` binary (the app finds `shuvoice` beside itself,
never via `PATH`), then open it with `shuvoice settings`.

## Develop

```bash
cargo build -p shuvoice-cli                 # bridge used by the app
SHUVOICE_BIN=../../target/debug/shuvoice bun run dev
bun run typecheck && bun run test
bun src/e2e/smoke.ts                        # real window + bridge, isolated config
```

`SHUVOICE_SETTINGS_NO_RESTART=1` makes Apply save without restarting the
service (used by the smoke test).

## GPUI patch

GPUIX 0.10.0's GPUI binds every Wayland seat and keeps the last one. With extra
seats (e.g. `cua-hyprland-plugin`'s `Cua-Agent`), the window never receives
physical input. `patches/gpui-primary-seat.patch` keeps the first (primary)
seat; `scripts/build-addon.sh` applies it and builds the addon. Upgrade GPUIX
and the addon together, and re-check the patch on upgrade.
