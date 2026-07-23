"""Protocol errors (never embed transcripts or secrets in messages)."""

from __future__ import annotations


class ProtocolError(Exception):
    """Framing / handshake / control-plane failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"
