"""CLI entry point for shuvoice."""

from __future__ import annotations

from .cli import apply_cli_overrides as _apply_cli_overrides
from .cli import main as _cli_main
from .cli.commands.preflight import run_preflight as _run_preflight
from .cli.commands.wizard import run_welcome_wizard as _run_welcome_wizard

__all__ = [
    "_apply_cli_overrides",
    "_run_preflight",
    "_run_welcome_wizard",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
