#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for Waybar custom module commands.
#
# It prefers the installed console script (`shuvoice-waybar`) and falls back
# to the repo virtualenv/module path for local development checkouts.

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  SELF="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || printf '%s' "$SCRIPT_PATH")"
else
  SELF="$SCRIPT_PATH"
fi
ROOT_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"

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

for venv in "$ROOT_DIR/.venv312" "$ROOT_DIR/.venv"; do
  if [ -x "$venv/bin/shuvoice-waybar" ]; then
    exec "$venv/bin/shuvoice-waybar" "$@"
  fi
done

for venv in "$ROOT_DIR/.venv312" "$ROOT_DIR/.venv"; do
  if [ -x "$venv/bin/python" ]; then
    exec "$venv/bin/python" -m shuvoice.waybar "$@"
  fi
done

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m shuvoice.waybar "$@"
fi

exec python -m shuvoice.waybar "$@"
