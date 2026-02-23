#!/usr/bin/env bash
set -euo pipefail

# Remove a previously installed Waybar wrapper symlink.
#
# Default remove target:
#   ~/.local/bin/shuvoice-waybar
#
# Safety behavior:
# - If the target is not a symlink to this repo's wrapper, the script aborts
#   unless --force is provided.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_SCRIPT="$ROOT_DIR/scripts/shuvoice-waybar.sh"

BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
LINK_NAME="shuvoice-waybar"
FORCE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: uninstall-waybar-wrapper.sh [options]

Options:
  --bin-dir <dir>    Install directory (default: $XDG_BIN_HOME or ~/.local/bin)
  --name <name>      Symlink name (default: shuvoice-waybar)
  --force            Remove destination even if it is not this repo's symlink
  --dry-run          Print planned actions without changing anything
  -h, --help         Show this help text
EOF
}

resolve_path() {
  local path="$1"
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$path" 2>/dev/null || printf '%s' "$path"
  else
    printf '%s' "$path"
  fi
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-dir)
      [[ $# -ge 2 ]] || die "--bin-dir requires a value"
      BIN_DIR="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      LINK_NAME="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$LINK_NAME" ]] || die "Link name must not be empty"

DEST_PATH="$BIN_DIR/$LINK_NAME"
SOURCE_REAL="$(resolve_path "$SOURCE_SCRIPT")"

if [[ ! -e "$DEST_PATH" && ! -L "$DEST_PATH" ]]; then
  log "Nothing to remove: $DEST_PATH"
  exit 0
fi

if [[ -L "$DEST_PATH" ]]; then
  DEST_REAL="$(resolve_path "$DEST_PATH")"
  if [[ "$DEST_REAL" != "$SOURCE_REAL" && "$FORCE" -ne 1 ]]; then
    die "$DEST_PATH points to $DEST_REAL (expected $SOURCE_REAL). Use --force to remove anyway."
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would remove symlink: $DEST_PATH -> $DEST_REAL"
    exit 0
  fi

  rm -f "$DEST_PATH"
  log "Removed symlink: $DEST_PATH"
  exit 0
fi

if [[ "$FORCE" -ne 1 ]]; then
  die "$DEST_PATH exists but is not a symlink. Use --force to remove it."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Would remove file: $DEST_PATH"
  exit 0
fi

rm -f "$DEST_PATH"
log "Removed file: $DEST_PATH"
