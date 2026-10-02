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

`shuvoice settings` opens the full settings window. Run
`shuvoice-settings --onboarding` for Engine → Microphone → Read-aloud → Shortcut
→ Finish, or skip directly to settings. `shuvoice wizard` remains the setup
entry point. The desktop launcher uses `shuvoice settings`.

Search finds fields across sections; less common fields live under Advanced.
Apply & Restart validates and waits for dictation/read-aloud to finish, reserves
the service, saves a patch with a backup, then restarts. Cancel is available
while waiting. A changed file requires Reload; saved-but-not-ready results are
shown separately. Missing models warn without blocking Apply. Shortcut changes
use Preview → Confirm and apply immediately, outside the settings draft.

Optional pages feature-detect the bridge; older binaries omit model downloads,
hint capability/preview and shortcut editing. Setup completion needs the new
bridge. No defaults or config validation are duplicated in TypeScript.

```bash
cargo build -p shuvoice-cli                 # bridge used by the app
SHUVOICE_BIN=../../target/debug/shuvoice bun run dev
bun run typecheck && bun run test
bun src/e2e/smoke.ts                        # real window + bridge, isolated config
bun src/e2e/features.ts                     # real bridge: schema, features, onboarding
```

`SHUVOICE_SETTINGS_NO_RESTART=1` makes Apply save without restarting the
service (used by the smoke test).
Both real-window tests isolate `HOME`, config, data, cache, runtime and D-Bus
paths. Only `WAYLAND_DISPLAY` points to the real compositor. Shortcut dry-runs
use a private Hyprland fixture; tests never confirm a shortcut or download models.
The features test checks all schema fields and completes onboarding with restart
disabled. Run it against a matching bridge built with `cargo build -p shuvoice-cli`.

Set `SHUVOICE_SETTINGS_SHOTS=/tmp/shuvcode/frontend-shots-v2` on the smoke test to
capture each page with `grim`. The features test captures each page and onboarding
step against the real bridge in that directory by default. Both screenshot paths use `hyprctl` window geometry and
`HYPRLAND_INSTANCE_SIGNATURE` (falling back to the test desktop signature in
`/tmp/shuvcode/pr69/live/hypr.sig`).

## GPUI patch

GPUIX 0.10.0's GPUI binds every Wayland seat and keeps the last one. With extra
seats (e.g. `cua-hyprland-plugin`'s `Cua-Agent`), the window never receives
physical input. `patches/gpui-primary-seat.patch` keeps the first (primary)
seat; `scripts/build-addon.sh` applies it and builds the addon. Upgrade GPUIX
and the addon together, and re-check the patch on upgrade.

For a pre-fetched source tree, run `bash scripts/build-addon.sh /path/to/gpuix`.
It requires tag `@gpuix/react@0.10.0` and Zed commit
`81c99f816b4a5f69d3c014774068034c24d1d7af`. The Arch recipe fetches both sources
and applies the same patch; the compiled executable embeds the addon.

If the bridge cannot be found, set `SHUVOICE_BIN=/absolute/path/to/shuvoice`.
If the window ignores physical input on a multi-seat desktop, rebuild the
patched addon and executable rather than using the stock npm native binary.

## Branding

`assets/` holds images derived from `docs/assets/branding/`: `splash.png` is the
dark lockup cropped with its edges faded to transparent, and `logo-lockup.png`
is the transparent lockup cropped to its content. Both are embedded in the
compiled binary (`src/branding.ts`).
