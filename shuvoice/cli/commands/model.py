"""Model management commands."""

from __future__ import annotations

import sys

from ...asr import get_backend_class
from ...config import Config


def download_model(config: Config) -> int:
    backend_cls = get_backend_class(config.asr_backend)

    try:
        kwargs: dict[str, object] = {}
        if config.asr_backend == "nemo":
            kwargs["model_name"] = config.model_name
        elif config.asr_backend == "sherpa":
            kwargs["model_name"] = config.sherpa_model_name
            kwargs["model_dir"] = config.sherpa_model_dir

        backend_cls.download_model(**kwargs)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Model downloaded successfully.")
    return 0
