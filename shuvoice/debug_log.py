"""In-process log capture for the debug overlay."""

from __future__ import annotations

import logging
import threading
from collections import deque


class RecentLogBuffer(logging.Handler):
    """Thread-safe ring buffer of formatted log lines."""

    def __init__(self, *, max_entries: int = 400):
        super().__init__(level=logging.DEBUG)
        self._lock = threading.Lock()
        self._entries: deque[str] = deque(maxlen=max(1, int(max_entries)))
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            try:
                line = f"{record.levelname} {record.name}: {record.getMessage()}"
            except Exception:
                line = f"{record.levelname} {record.name}: <unformattable log record>"

        with self._lock:
            self._entries.append(line)

    def tail(self, *, max_lines: int = 12) -> list[str]:
        with self._lock:
            if max_lines <= 0:
                return []
            return list(self._entries)[-int(max_lines) :]

    def render(self, *, max_lines: int = 12) -> str:
        return "\n".join(self.tail(max_lines=max_lines))
