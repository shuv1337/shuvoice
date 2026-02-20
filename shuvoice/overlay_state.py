"""Pure overlay state definitions (headless-test friendly)."""

from __future__ import annotations

OVERLAY_STATE_LISTENING = "listening"
OVERLAY_STATE_PROCESSING = "processing"
OVERLAY_STATE_ERROR = "error"

OVERLAY_STATE_CLASSES = {
    OVERLAY_STATE_LISTENING: "state-listening",
    OVERLAY_STATE_PROCESSING: "state-processing",
    OVERLAY_STATE_ERROR: "state-error",
}


def overlay_state_class(state: str) -> str:
    try:
        return OVERLAY_STATE_CLASSES[state]
    except KeyError as exc:
        allowed = ", ".join(sorted(OVERLAY_STATE_CLASSES))
        raise ValueError(f"Unknown overlay state '{state}'. Expected one of: {allowed}") from exc
