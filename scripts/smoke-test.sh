#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== ShuVoice smoke test =="

echo
printf "[1/4] Python compile check... "
python -m compileall shuvoice >/dev/null
printf "ok\n"

echo
printf "[2/4] CLI help check... "
python -m shuvoice --help >/dev/null
printf "ok\n"

echo
printf "[3/4] Preflight check...\n"
if python -m shuvoice --preflight; then
  echo "Preflight: READY"
else
  echo "Preflight: NOT READY (see output above)"
fi

echo
printf "[4/4] Control socket sanity (expected to fail if app not running)...\n"
if python -m shuvoice --control ping; then
  echo "Control socket reachable"
else
  echo "Control socket not reachable (start app first to test IPC path)"
fi

cat <<'EOF'

Manual E2E checklist:

1) Start shuvoice:
   python -m shuvoice --output-mode final_only

2) Verify evdev mode:
   - Press/hold configured hotkey, speak, release
   - Confirm overlay appears and final text is injected

3) Verify IPC mode fallback:
   - Start app with: python -m shuvoice --hotkey-backend ipc
   - In another terminal:
       python -m shuvoice --control start
       python -m shuvoice --control status
       python -m shuvoice --control stop
   - Confirm recording toggles and status transitions

4) Hyprland bind/bindr integration test:
   bind  = , F9,  exec, shuvoice --control start
   bindr = , F9,  exec, shuvoice --control stop

5) Streaming partial mode:
   python -m shuvoice --output-mode streaming_partial
   - Confirm partial text replacement works and final commit is clean

EOF
