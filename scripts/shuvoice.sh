#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper so `shuvoice` works from PATH during development.
#
# Resolution order:
#   1. Installed console script (pip install -e .) if it isn't this wrapper
#   2. Repo virtualenv console script
#   3. Repo virtualenv `python -m shuvoice`
#   4. System python -m shuvoice

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

if command -v shuvoice >/dev/null 2>&1; then
  CMD_PATH="$(command -v shuvoice)"
  if [ "$(resolved_path "$CMD_PATH")" != "$SELF" ]; then
    exec "$CMD_PATH" "$@"
  fi
fi

# Prefer .venv312 (has all backends installed), fall back to .venv.
for venv in "$ROOT_DIR/.venv312" "$ROOT_DIR/.venv"; do
  if [ -x "$venv/bin/shuvoice" ]; then
    exec "$venv/bin/shuvoice" "$@"
  fi
done

for venv in "$ROOT_DIR/.venv312" "$ROOT_DIR/.venv"; do
  if [ -x "$venv/bin/python" ]; then
    exec "$venv/bin/python" -m shuvoice "$@"
  fi
done

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m shuvoice "$@"
fi

exec python -m shuvoice "$@"
