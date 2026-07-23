"""Stdio framed transport helpers."""

from __future__ import annotations

import sys
from typing import BinaryIO

from .framing import Frame, read_frame, write_frame


class FramedStdio:
    """Bidirectional framed connection, typically process stdin/stdout."""

    def __init__(self, reader: BinaryIO | None = None, writer: BinaryIO | None = None) -> None:
        self.reader = reader if reader is not None else sys.stdin.buffer
        self.writer = writer if writer is not None else sys.stdout.buffer

    def read_frame(self) -> Frame:
        return read_frame(self.reader)

    def write_frame(self, frame: Frame) -> None:
        write_frame(self.writer, frame)

    def write_json(self, obj: dict) -> None:
        self.write_frame(Frame.json_bytes(obj))
