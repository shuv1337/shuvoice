"""Entry: python -m melotts [--fake] [--device auto|cpu|cuda] [-v]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WORKERS_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKERS_ROOT))

from shuvoice_worker_proto.server import run_worker  # noqa: E402

from .worker import MeloTtsHandler  # noqa: E402


def _device_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return os.environ.get("SHUVOICE_MELOTTS_DEVICE", "auto")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fake = "--fake" in argv or os.environ.get("SHUVOICE_WORKER_FAKE") == "1"
    device = _device_from_argv(argv)

    def factory() -> MeloTtsHandler:
        return MeloTtsHandler(fake=fake, device=device)

    return run_worker(factory, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
