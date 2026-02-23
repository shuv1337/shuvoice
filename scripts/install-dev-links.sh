#!/usr/bin/env bash
set -euo pipefail

# Install development symlinks so `shuvoice` and `shuvoice-waybar` are
# available on PATH without a full pip install.
#
# Default:
#   ~/.local/bin/shuvoice        -> <repo>/scripts/shuvoice.sh
#   ~/.local/bin/shuvoice-waybar -> <repo>/scripts/shuvoice-waybar.sh
#
# Supports --bin-dir, --dry-run, and --force.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
FORCE=0
DRY_RUN=0

# name -> source script (relative to SCRIPT_DIR)
LINKS=(
  "shuvoice:shuvoice.sh"
  "shuvoice-waybar:shuvoice-waybar.sh"
)

usage() {
  cat <<'EOF'
Usage: install-dev-links.sh [options]

Install development symlinks for shuvoice and shuvoice-waybar.

Options:
  --bin-dir <dir>    Install directory (default: $XDG_BIN_HOME or ~/.local/bin)
  --force            Replace existing files/symlinks at destination
  --dry-run          Print planned actions without changing anything
  -h, --help         Show this help text
EOF
}

resolve_path() {
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$1" 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-dir) [[ $# -ge 2 ]] || die "--bin-dir requires a value"; BIN_DIR="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         usage; die "Unknown argument: $1" ;;
  esac
done

install_link() {
  local link_name="$1" source_script="$2"
  local source_real dest_path

  source_real="$(resolve_path "$source_script")"
  dest_path="$BIN_DIR/$link_name"

  [[ -x "$source_script" ]] || die "Source not executable: $source_script"

  if [[ -L "$dest_path" ]]; then
    local dest_real
    dest_real="$(resolve_path "$dest_path")"
    if [[ "$dest_real" == "$source_real" ]]; then
      log "  ✓ $link_name (already correct)"
      return
    fi
    if [[ "$FORCE" -ne 1 ]]; then
      die "$dest_path points to $dest_real (use --force to replace)"
    fi
  elif [[ -e "$dest_path" ]]; then
    if [[ "$FORCE" -ne 1 ]]; then
      die "$dest_path already exists (use --force to replace)"
    fi
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  → would link $dest_path -> $source_real"
    return
  fi

  ln -sfn "$source_real" "$dest_path"
  log "  ✓ $link_name -> $source_real"
}

log "Installing dev symlinks into $BIN_DIR"

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$BIN_DIR"
fi

for entry in "${LINKS[@]}"; do
  link_name="${entry%%:*}"
  source_file="$SCRIPT_DIR/${entry#*:}"
  install_link "$link_name" "$source_file"
done

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
  log ""
  log "Note: $BIN_DIR is not in PATH. Add it to your shell profile."
fi
