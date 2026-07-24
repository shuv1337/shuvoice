#!/usr/bin/env bash
# Headless smoke checks for the Rust ShuVoice cutover.
#
# Default mode is safe for CI and local machines:
#   - no systemctl start/stop/restart
#   - no live mic capture
#   - no display/GUI launch
#   - no implicit model/network downloads
#
# Optional live checks are printed at the end for manual runs only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== ShuVoice smoke test (safe / headless) =="

resolve_shuvoice() {
  if [ -x "$ROOT_DIR/target/release/shuvoice" ]; then
    echo "$ROOT_DIR/target/release/shuvoice"
    return
  fi
  if [ -x "$ROOT_DIR/target/debug/shuvoice" ]; then
    echo "$ROOT_DIR/target/debug/shuvoice"
    return
  fi
  if [ -x "$ROOT_DIR/scripts/shuvoice.sh" ]; then
    echo "$ROOT_DIR/scripts/shuvoice.sh"
    return
  fi
  if command -v shuvoice >/dev/null 2>&1; then
    command -v shuvoice
    return
  fi
  return 1
}

resolve_waybar() {
  if [ -x "$ROOT_DIR/target/release/shuvoice-waybar" ]; then
    echo "$ROOT_DIR/target/release/shuvoice-waybar"
    return
  fi
  if [ -x "$ROOT_DIR/target/debug/shuvoice-waybar" ]; then
    echo "$ROOT_DIR/target/debug/shuvoice-waybar"
    return
  fi
  if [ -x "$ROOT_DIR/scripts/shuvoice-waybar.sh" ]; then
    echo "$ROOT_DIR/scripts/shuvoice-waybar.sh"
    return
  fi
  if command -v shuvoice-waybar >/dev/null 2>&1; then
    command -v shuvoice-waybar
    return
  fi
  return 1
}

if ! SHUVOICE="$(resolve_shuvoice)"; then
  echo "ERROR: no shuvoice binary found." >&2
  echo "Build with: cargo build -p shuvoice-cli --features desktop" >&2
  exit 1
fi

WAYBAR_BIN=""
if WAYBAR_BIN="$(resolve_waybar)"; then
  :
else
  WAYBAR_BIN=""
fi

echo "Using shuvoice: $SHUVOICE"
if [ -n "$WAYBAR_BIN" ]; then
  echo "Using shuvoice-waybar: $WAYBAR_BIN"
fi

step=0
total=6

# Run a named check. Command stdout/stderr are suppressed on success.
run_step() {
  local title="$1"
  shift
  step=$((step + 1))
  echo
  printf '[%d/%d] %s... ' "$step" "$total" "$title"
  local out_file err_file rc
  out_file="$(mktemp)"
  err_file="$(mktemp)"
  set +e
  "$@" >"$out_file" 2>"$err_file"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    printf 'ok\n'
    rm -f "$out_file" "$err_file"
    return 0
  fi
  printf 'FAILED (exit %s)\n' "$rc"
  if [ -s "$out_file" ]; then
    echo "--- stdout ---" >&2
    cat "$out_file" >&2
  fi
  if [ -s "$err_file" ]; then
    echo "--- stderr ---" >&2
    cat "$err_file" >&2
  fi
  rm -f "$out_file" "$err_file"
  return "$rc"
}

run_step "CLI help" \
  "$SHUVOICE" --help

run_step "Config path" \
  "$SHUVOICE" config path

run_step "Config validate" \
  "$SHUVOICE" config validate

run_step "Worker tree layout" \
  bash -c '
    set -euo pipefail
    root="'"$ROOT_DIR"'/workers"
    for pkg in shuvoice_worker_proto nemo_asr moonshine_asr melotts; do
      test -f "$root/$pkg/__init__.py"
      if [ "$pkg" != "shuvoice_worker_proto" ]; then
        test -f "$root/$pkg/__main__.py"
      fi
    done
    grep -q "/usr/lib/shuvoice/workers" \
      "'"$ROOT_DIR"'/crates/shuvoice-cli/src/compose/worker_runtime.rs"
  '

if command -v python3 >/dev/null 2>&1; then
  run_step "Worker stdlib unittest" \
    bash -c 'cd "'"$ROOT_DIR"'/workers" && python3 -m unittest discover -s tests -q'
else
  run_step "Worker stdlib unittest" \
    bash -c 'echo "python3 missing" >&2; exit 1'
fi

if [ -n "$WAYBAR_BIN" ]; then
  run_step "Waybar helper help" \
    "$WAYBAR_BIN" --help
else
  run_step "Waybar helper help" \
    bash -c 'echo "shuvoice-waybar binary missing" >&2; exit 1'
fi

cat <<'EOT'

Safe smoke checks passed.

Optional live checks (manual only — may touch service, mic, or display):

1) Build + run desktop binary:
   cargo run -p shuvoice-cli --features desktop --bin shuvoice

2) IPC against a running instance (starts nothing itself):
   shuvoice control ping --control-wait-sec 0
   shuvoice control status --control-wait-sec 0

3) Preflight on real hardware (may enumerate audio devices):
   shuvoice preflight

4) systemd user unit (mutates service state — do not run in CI):
   systemctl --user restart shuvoice.service
   systemctl --user show -p ExecMainStatus -p NRestarts shuvoice.service
   # Exit 78 must not restart-loop (RestartPreventExitStatus=78).

5) Hyprland PTT bind smoke:
   bind  = , F9,  exec, shuvoice control start --control-wait-sec 0
   bindr = , F9,  exec, shuvoice control stop --control-wait-sec 0

6) Optional worker runtime (requires Python engine venv via setup):
   export SHUVOICE_WORKERS_DIR="$PWD/workers"
   shuvoice setup --install-missing --asr-backend moonshine
EOT
