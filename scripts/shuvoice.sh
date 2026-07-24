#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper so `shuvoice` works from PATH during development.
#
# Resolution order (Rust-only; no Python/.venv fallbacks):
#   1. Repo Cargo release binary
#   2. Repo Cargo debug binary
#   3. Installed binary on PATH (if it isn't this wrapper)

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

for cand in \
  "$ROOT_DIR/target/release/shuvoice" \
  "$ROOT_DIR/target/debug/shuvoice"
do
  if [ -x "$cand" ]; then
    exec "$cand" "$@"
  fi
done

if command -v shuvoice >/dev/null 2>&1; then
  CMD_PATH="$(command -v shuvoice)"
  if [ "$(resolved_path "$CMD_PATH")" != "$SELF" ]; then
    exec "$CMD_PATH" "$@"
  fi
fi

printf 'shuvoice: no Rust binary found.\n' >&2
printf 'Build one with: cargo build -p shuvoice-cli --features desktop\n' >&2
printf 'Or install the package providing /usr/bin/shuvoice.\n' >&2
exit 127
