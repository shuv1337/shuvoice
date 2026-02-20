"""Text injection via wtype and wl-clipboard."""

from __future__ import annotations

import logging
import subprocess
import time

log = logging.getLogger(__name__)

_BACKSPACE_BATCH_SIZE = 50


class StreamingTyper:
    """Inject text into the focused Wayland window.

    Strategy:
    - streaming partials: backspace previous partial + type new partial via wtype
    - final text: clipboard paste (wl-copy + Ctrl+V) for robustness
    - fallback: direct wtype typing when clipboard paste fails
    """

    def __init__(
        self,
        preserve_clipboard: bool = False,
        retry_attempts: int = 2,
        retry_delay_ms: int = 40,
    ):
        self.last_partial_len = 0
        self.preserve_clipboard = preserve_clipboard
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay_s = max(0.0, retry_delay_ms / 1000.0)

    def _run(self, args: list[str], op: str, attempts: int | None = None) -> bool:
        attempts = attempts if attempts is not None else self.retry_attempts
        attempts = max(1, attempts)

        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(args, check=True, timeout=5)
                return True
            except (subprocess.SubprocessError, OSError) as e:
                if attempt == attempts:
                    log.error("%s failed after %d attempt(s): %s", op, attempts, e)
                    return False
                log.warning("%s attempt %d/%d failed: %s", op, attempt, attempts, e)
                if self.retry_delay_s:
                    time.sleep(self.retry_delay_s)

        return False

    @staticmethod
    def _backspace_args(count: int) -> list[str]:
        args = ["wtype"]
        for _ in range(count):
            args.extend(["-k", "BackSpace"])
        return args

    def _send_backspaces(self, count: int, op: str) -> bool:
        if count <= 0:
            return True

        remaining = count
        while remaining > 0:
            batch = min(remaining, _BACKSPACE_BATCH_SIZE)
            ok = self._run(self._backspace_args(batch), op)
            if not ok:
                return False
            remaining -= batch
        return True

    def _backspace_partial(self) -> bool:
        return self._send_backspaces(self.last_partial_len, "wtype backspace")

    def _type_direct(self, text: str) -> bool:
        if not text:
            return True
        return self._run(["wtype", "--", text], "wtype direct type")

    def _paste_via_clipboard(self, text: str) -> bool:
        if not text:
            return True

        copied = self._run(["wl-copy", "--", text], "wl-copy set")
        if not copied:
            return False

        return self._run(
            ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
            "wtype ctrl+v",
        )

    def _capture_clipboard(self) -> tuple[bool, str]:
        """Return (had_content, content). Best effort only."""
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                check=True,
                timeout=3,
                capture_output=True,
                text=True,
            )
            return True, result.stdout
        except Exception as e:
            log.debug("Could not capture clipboard for preservation: %s", e)
            return False, ""

    def _restore_clipboard(self, had_content: bool, content: str):
        if not self.preserve_clipboard:
            return

        if had_content:
            self._run(["wl-copy", "--", content], "wl-copy restore", attempts=1)
        else:
            self._run(["wl-copy", "--clear"], "wl-copy clear", attempts=1)

    def update_partial(self, new_text: str):
        """Replace previous partial text with new partial text."""
        if not new_text and self.last_partial_len == 0:
            return

        backspaced = self._send_backspaces(self.last_partial_len, "wtype partial backspace")
        if not backspaced:
            self.last_partial_len = 0
            return

        if new_text:
            typed = self._run(["wtype", "--", new_text], "wtype partial type")
            if typed:
                self.last_partial_len = len(new_text)
            else:
                self.last_partial_len = 0
            return

        self.last_partial_len = 0

    def commit_final(self, final_text: str):
        """Erase partial text, then paste final text (with direct-typing fallback)."""
        had_clip = False
        clip_content = ""
        if self.preserve_clipboard:
            had_clip, clip_content = self._capture_clipboard()

        self._backspace_partial()

        if final_text:
            pasted = self._paste_via_clipboard(final_text)
            if not pasted:
                log.warning("Clipboard paste failed, falling back to direct typing")
                self._type_direct(final_text)

        self._restore_clipboard(had_clip, clip_content)
        self.last_partial_len = 0

    def reset(self):
        """Reset tracking state without sending any keystrokes."""
        self.last_partial_len = 0
