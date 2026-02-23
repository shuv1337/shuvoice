#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for Waybar custom module commands.
#
# It prefers the installed console script (`shuvoice-waybar`) and falls back
# to the repo virtualenv/module path for local development checkouts.

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolved_path() {
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$1" 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}

if command -v shuvoice-waybar >/dev/null 2>&1; then
  CMD_PATH="$(command -v shuvoice-waybar)"
  if [ "$(resolved_path "$CMD_PATH")" != "$SELF" ]; then
    exec "$CMD_PATH" "$@"
  fi
fi

if [ -x "$ROOT_DIR/.venv/bin/shuvoice-waybar" ]; then
  exec "$ROOT_DIR/.venv/bin/shuvoice-waybar" "$@"
fi

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  exec "$ROOT_DIR/.venv/bin/python" -m shuvoice.waybar "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m shuvoice.waybar "$@"
fi

exec python -m shuvoice.waybar "$@"
