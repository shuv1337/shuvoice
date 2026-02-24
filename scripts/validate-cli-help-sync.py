#!/usr/bin/env python3
"""Validate CLI help contains expected subcommand groups."""

from __future__ import annotations

import subprocess
import sys

REQUIRED_HELP_TOKENS = [
    "run",
    "control",
    "preflight",
    "wizard",
    "config",
    "model",
    "audio",
]


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "shuvoice", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or "failed to run shuvoice --help")
        return 1

    help_text = proc.stdout
    missing = [token for token in REQUIRED_HELP_TOKENS if token not in help_text]
    if missing:
        print("CLI help is missing expected tokens: " + ", ".join(missing))
        return 1

    print("CLI help sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
