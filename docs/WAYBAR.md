# Waybar Integration

`shuvoice-waybar` is a **Rust** helper binary (same `shuvoice-cli` package as
`shuvoice`). It emits JSON for a Waybar `custom/shuvoice` module: recording
state, service state, configured TTS provider, default voice, and keybind hints.

## Module

### Omarchy Quickshell bar

Recent Omarchy desktops use Quickshell instead of Waybar. Install the bundled
widget to display ShuVoice in that bar (both `shuvoice` and `shuvoice-waybar`
must be on the shell's `PATH`):

```bash
mkdir -p ~/.config/omarchy/plugins/shuv.shuvoice
cp packaging/omarchy/shuv.shuvoice/{manifest.json,Widget.qml} ~/.config/omarchy/plugins/shuv.shuvoice/
omarchy-shell shell rescanPlugins
```

Discovery is asynchronous. Once `omarchy plugin list` shows `shuv.shuvoice`,
enable it:

```bash
omarchy plugin enable shuv.shuvoice
```

The widget uses the same live status helper. Left-click toggles recording,
middle-click toggles the service, and right-click opens setup. Hover for status
and configured backend details. It follows the active Omarchy bar theme.

### Waybar

Prefer the installed binary on `PATH` (AUR / packaging):

```jsonc
"custom/shuvoice": {
  "return-type": "json",
  "exec": "shuvoice-waybar status",
  "interval": 1,
  "on-click": "shuvoice-waybar toggle-record",
  "on-click-middle": "shuvoice-waybar service-toggle",
  "on-click-right": "shuvoice-waybar menu",
  "tooltip": true
}
```

## CSS

```css
#custom-shuvoice.recording  { color: #f38ba8; }
#custom-shuvoice.processing { color: #fab387; }
#custom-shuvoice.idle       { color: #a6e3a1; }
#custom-shuvoice.starting   { color: #f9e2af; }
#custom-shuvoice.stopped    { color: #7f849c; }
#custom-shuvoice.error      { color: #f38ba8; }
```

## Commands

```bash
shuvoice-waybar status
shuvoice-waybar menu
shuvoice-waybar toggle-record
shuvoice-waybar service-toggle
shuvoice-waybar launch-wizard
```

The right-click menu uses the first available launcher from
`omarchy-launch-walker`, `walker`, `wofi`, `rofi`, `bemenu`, or `dmenu`.

## Finding the binary

Resolution order for a working helper:

1. **Installed package**: `/usr/bin/shuvoice-waybar` (on `PATH` after AUR install)
2. **Source build**: `target/release/shuvoice-waybar` or `target/debug/shuvoice-waybar`
   after `cargo build -p shuvoice-cli` (default `desktop` features)
3. **Repo launcher script** (optional): `scripts/shuvoice-waybar.sh` — resolves
   `target/{release,debug}/shuvoice-waybar`, then a real PATH binary. **Rust only**;
   no Python / `.venv` fallback.

If Waybar cannot find `shuvoice-waybar` on PATH, either:

```bash
# Install a ~/.local/bin wrapper that prefers the repo Rust binary
./scripts/install-waybar-wrapper.sh
```

or point Waybar `exec` / click actions at an absolute path:

```text
/path/to/shuvoice/target/release/shuvoice-waybar
```

or the launcher script:

```text
/path/to/shuvoice/scripts/shuvoice-waybar.sh
```

Complete examples:

- [`../examples/waybar-custom-shuvoice.jsonc`](../examples/waybar-custom-shuvoice.jsonc) — installed / PATH binary
- [`../examples/waybar-custom-shuvoice-wrapper.jsonc`](../examples/waybar-custom-shuvoice-wrapper.jsonc) — repo launcher script
- [`../examples/waybar-shuvoice.css`](../examples/waybar-shuvoice.css)
