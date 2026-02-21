from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock sounddevice/evdev/gi to prevent import errors in CI/headless environments
if "sounddevice" not in sys.modules:
    mock_sd = MagicMock()
    mock_sd.PortAudioError = OSError
    sys.modules["sounddevice"] = mock_sd

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]):
    for item in items:
        if "gui" not in item.keywords:
            item.add_marker(pytest.mark.unit)
