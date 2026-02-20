from __future__ import annotations

import pytest

from shuvoice.overlay_state import (
    OVERLAY_STATE_ERROR,
    OVERLAY_STATE_LISTENING,
    OVERLAY_STATE_PROCESSING,
    overlay_state_class,
)


def test_overlay_state_class_valid_states():
    assert overlay_state_class(OVERLAY_STATE_LISTENING) == "state-listening"
    assert overlay_state_class(OVERLAY_STATE_PROCESSING) == "state-processing"
    assert overlay_state_class(OVERLAY_STATE_ERROR) == "state-error"


def test_overlay_state_class_rejects_unknown_state():
    with pytest.raises(ValueError, match="Unknown overlay state"):
        overlay_state_class("unknown")
