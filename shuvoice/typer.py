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
    - final text: mode-driven (`auto`, `clipboard`, `direct`)
    - clipboard path fallback: direct wtype typing when paste fails
    """

    def __init__(
        self,
        final_injection_mode: str = "auto",
        preserve_clipboard: bool = False,
        clipboard_settle_delay_ms: int = 40,
        retry_attempts: int = 2,
        retry_delay_ms: int = 40,
    ):
        self.last_partial_len = 0
        self.last_partial_text = ""
        self.final_injection_mode = final_injection_mode
        self.preserve_clipboard = preserve_clipboard
        self.clipboard_settle_delay_s = max(0.0, clipboard_settle_delay_ms / 1000.0)
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay_s = max(0.0, retry_delay_ms / 1000.0)
        self._watchers_detected: bool | None = None
        self._watchers_last_checked_monotonic = 0.0
        self._watchers_cache_ttl_s = 30.0

    def _run(self, args: list[str], op: str, attempts: int | None = None) -> bool:
        attempts = attempts if attempts is not None else self.retry_attempts
        attempts = max(1, attempts)

        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(args, check=True, timeout=5)
                return True
            except (subprocess.SubprocessError, OSError) as e:
                # Sanitize error message to avoid leaking sensitive text
                err_msg = str(e)
                if isinstance(e, subprocess.SubprocessError):
                    cmd_name = args[0] if args else "subprocess"
                    if isinstance(e, subprocess.CalledProcessError):
                        err_msg = f"{cmd_name} failed with exit code {e.returncode}"
                    elif isinstance(e, subprocess.TimeoutExpired):
                        err_msg = f"{cmd_name} timed out after {e.timeout}s"

                if attempt == attempts:
                    log.error("%s failed after %d attempt(s): %s", op, attempts, err_msg)
                    return False
                log.warning("%s attempt %d/%d failed: %s", op, attempt, attempts, err_msg)
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

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        idx = 0
        while idx < limit and left[idx] == right[idx]:
            idx += 1
        return idx

    def _detect_clipboard_watchers(self) -> bool:
        """Best-effort detection of active clipboard managers/watchers."""
        now = time.monotonic()
        if self._watchers_detected is not None:
            age = now - self._watchers_last_checked_monotonic
            if age < self._watchers_cache_ttl_s:
                return self._watchers_detected

        try:
            # We look for common Wayland clipboard daemon command lines.
            result = subprocess.run(
                ["pgrep", "-a", "-f", "wl-paste --watch|wl-clip-persist|elephant"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # pgrep returns 0 if matches found, 1 if none found
            self._watchers_detected = result.returncode == 0
            if self._watchers_detected:
                log.info("Detected clipboard watcher(s), enabling direct final typing.")
        except Exception as e:
            log.debug("Failed to detect clipboard watchers: %s", e)
            self._watchers_detected = False

        self._watchers_last_checked_monotonic = now
        return self._watchers_detected

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

        if self.clipboard_settle_delay_s > 0:
            time.sleep(self.clipboard_settle_delay_s)

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
        """Replace previous partial text using a diff-based suffix update."""
        old_text = self.last_partial_text
        if not new_text and not old_text:
            return

        common_prefix = self._common_prefix_len(old_text, new_text)
        to_delete = len(old_text) - common_prefix
        to_insert = new_text[common_prefix:]

        if to_delete > 0:
            backspaced = self._send_backspaces(to_delete, "wtype partial backspace")
            if not backspaced:
                self.last_partial_len = 0
                self.last_partial_text = ""
                return

        if to_insert:
            typed = self._run(["wtype", "--", to_insert], "wtype partial type")
            if not typed:
                self.last_partial_len = 0
                self.last_partial_text = ""
                return

        self.last_partial_text = new_text
        self.last_partial_len = len(new_text)

    def commit_final(self, final_text: str):
        """Erase partial text, then inject final text using the resolved mode."""
        use_clipboard = True
        if self.final_injection_mode == "direct":
            use_clipboard = False
        elif self.final_injection_mode == "auto":
            use_clipboard = not self._detect_clipboard_watchers()

        if not use_clipboard:
            # Efficient suffix update for direct mode
            self.update_partial(final_text)
            self.reset()
            return

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
        self.last_partial_text = ""

    def reset(self):
        """Reset tracking state without sending any keystrokes."""
        self.last_partial_len = 0
        self.last_partial_text = ""
