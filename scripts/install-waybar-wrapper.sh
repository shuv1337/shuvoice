#!/usr/bin/env bash
set -euo pipefail

# Install a convenient symlink for the Waybar wrapper helper so Waybar can
# call it from PATH without a repo-absolute path.
#
# Default install:
#   ~/.local/bin/shuvoice-waybar -> <repo>/scripts/shuvoice-waybar.sh
#
# Supports custom name/bin-dir plus --dry-run and --force.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_SCRIPT="$ROOT_DIR/scripts/shuvoice-waybar.sh"

BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
LINK_NAME="shuvoice-waybar"
FORCE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install-waybar-wrapper.sh [options]

Options:
  --bin-dir <dir>    Install directory (default: $XDG_BIN_HOME or ~/.local/bin)
  --name <name>      Symlink name (default: shuvoice-waybar)
  --force            Replace an existing file/symlink at destination
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
[[ -x "$SOURCE_SCRIPT" ]] || die "Source wrapper not executable: $SOURCE_SCRIPT"

DEST_PATH="$BIN_DIR/$LINK_NAME"
SOURCE_REAL="$(resolve_path "$SOURCE_SCRIPT")"

if [[ -L "$DEST_PATH" ]]; then
  DEST_REAL="$(resolve_path "$DEST_PATH")"
  if [[ "$DEST_REAL" == "$SOURCE_REAL" ]]; then
    log "Already installed: $DEST_PATH -> $SOURCE_REAL"
    exit 0
  fi
  if [[ "$FORCE" -ne 1 ]]; then
    die "$DEST_PATH already exists and points to $DEST_REAL (use --force to replace)"
  fi
elif [[ -e "$DEST_PATH" ]]; then
  if [[ "$FORCE" -ne 1 ]]; then
    die "$DEST_PATH already exists (use --force to replace)"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Would create directory: $BIN_DIR"
  log "Would install symlink: $DEST_PATH -> $SOURCE_REAL"
  exit 0
fi

mkdir -p "$BIN_DIR"
ln -sfn "$SOURCE_REAL" "$DEST_PATH"

log "Installed: $DEST_PATH -> $SOURCE_REAL"

if ! command -v "$LINK_NAME" >/dev/null 2>&1; then
  log "Note: '$LINK_NAME' is not currently in PATH."
  log "Add '$BIN_DIR' to PATH, then restart Waybar/session."
fi
