#!/usr/bin/env bash
set -euo pipefail

EXCLUDE_PATHS='^(docs/internal/|scripts/check-env-hardcoding\.sh$|PLAN-open-source-cleanup\.md$)'
ALLOWLIST_LINE_PATTERNS='(/usr/bin/shuvoice|/dev/input/event\*?|/path/to/|\$XDG_RUNTIME_DIR|https://github\.com/shuv1337|github\.com/shuv1337)'

mapfile -t files < <(git ls-files | rg -v "$EXCLUDE_PATHS" || true)

if [ ${#files[@]} -eq 0 ]; then
  echo "OK: no tracked files to scan"
  exit 0
fi

hits=$(rg -n --no-heading --color=never \
  -e '/home/' \
  -e '/run/user/1000' \
  -e '\.venv312' \
  -e 'dev\.shuv\.shuvoice' \
  "${files[@]}" || true)

if [ -z "$hits" ]; then
  echo "OK: no hardcoded environment values in public files"
  exit 0
fi

filtered_hits=$(printf '%s\n' "$hits" | rg -v "$ALLOWLIST_LINE_PATTERNS" || true)

if [ -n "$filtered_hits" ]; then
  echo "ERROR: hardcoded environment values found:"
  echo "$filtered_hits"
  exit 1
fi

echo "OK: no hardcoded environment values in public files"
