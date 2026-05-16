# Waybar Integration

`shuvoice-waybar` emits JSON for a Waybar `custom/shuvoice` module. It reports
recording state, service state, configured TTS provider, default voice, and
keybind hints.

## Module

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

If Waybar cannot find `shuvoice-waybar`, install the wrapper:

```bash
./scripts/install-waybar-wrapper.sh
```

Or point Waybar directly at the venv executable.

Complete examples:

- [`../examples/waybar-custom-shuvoice.jsonc`](../examples/waybar-custom-shuvoice.jsonc)
- [`../examples/waybar-shuvoice.css`](../examples/waybar-shuvoice.css)
