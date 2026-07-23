#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Ensure Rust workspace compiles (default desktop features). Idempotent check.
if command -v cargo >/dev/null 2>&1; then
  cargo check -p shuvoice-cli
else
  echo "warning: cargo not on PATH; install a Rust toolchain matching workspace rust-version" >&2
fi

echo "Environment ready (Rust workspace). Optional workers: cd workers && python -m unittest discover -s tests -v"
