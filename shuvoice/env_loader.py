"""Environment variable loading helpers."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def local_dev_env_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "shuvoice" / "local.dev"


def load_local_dev_env(
    path: Path | str | None = None,
    *,
    override: bool = False,
) -> int:
    """Load environment variables from ``~/.config/shuvoice/local.dev``.

    File format supports:
    - comments starting with ``#``
    - empty lines
    - ``KEY=value`` and ``export KEY=value``

    Existing environment values are preserved by default unless
    ``override=True`` is provided.
    """
    env_path = Path(path).expanduser() if path is not None else local_dev_env_path()
    if not env_path.is_file():
        return 0

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("Could not read %s: %s", env_path, exc)
        return 0

    loaded = 0
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            log.warning("Ignoring invalid line %s:%d (missing '=')", env_path, line_no)
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not _ENV_KEY_RE.fullmatch(key):
            log.warning("Ignoring invalid environment key %r in %s:%d", key, env_path, line_no)
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if not override and key in os.environ:
            continue

        os.environ[key] = value
        loaded += 1

    return loaded
