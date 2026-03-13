#!/usr/bin/env bash
set -euo pipefail

cd /home/shuv/repos/shuvoice

# Install dev dependencies (idempotent)
uv sync --dev 2>/dev/null || true

echo "Environment ready."
