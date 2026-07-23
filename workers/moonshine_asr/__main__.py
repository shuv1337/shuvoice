"""Entry: python -m moonshine_asr [--fake] [-v]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WORKERS_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKERS_ROOT))

from shuvoice_worker_proto.server import run_worker  # noqa: E402

from .worker import MoonshineAsrHandler  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fake = "--fake" in argv or os.environ.get("SHUVOICE_WORKER_FAKE") == "1"

    def factory() -> MoonshineAsrHandler:
        return MoonshineAsrHandler(fake=fake)

    return run_worker(factory, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
