"""Shared CLI command helpers."""

from __future__ import annotations

import argparse

from ...config import Config
from ..parser import apply_cli_overrides


def load_effective_config(args: argparse.Namespace) -> Config:
    """Load config, apply CLI overrides, and re-validate."""
    config = Config.load()
    apply_cli_overrides(args, config)
    # Re-validate after CLI overrides mutate fields.
    config.__post_init__()
    return config
