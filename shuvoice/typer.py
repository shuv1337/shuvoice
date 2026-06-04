"""Text injection via wtype and wl-clipboard."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

_BACKSPACE_BATCH_SIZE = 50
_YDOTOOL_KEY_DELAY_MS = 0
_YDOTOOL_HOLD_DELAY_MS = 0
_KEY_BACKSPACE = 14
_KEY_LEFTCTRL = 29
_KEY_V = 47
_LINE_BREAK_RE = re.compile(r"[ \t\f\v]*(?:\r\n|\r|\n)+[ \t\f\v]*")


def sanitize_final_injection_text(text: str) -> str:
    """Return final STT text that is safe for Enter-to-submit prompt boxes."""
    if not text:
        return text
    return _LINE_BREAK_RE.sub(" ", text).strip()


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
        subprocess_timeout: float = 5.0,
    ):
        self.last_partial_len = 0
        self.last_partial_text = ""
        self.final_injection_mode = final_injection_mode
        self.preserve_clipboard = preserve_clipboard
        self.clipboard_settle_delay_s = max(0.0, clipboard_settle_delay_ms / 1000.0)
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay_s = max(0.0, retry_delay_ms / 1000.0)
        self.subprocess_timeout = max(1.0, float(subprocess_timeout))
        self._watchers_detected: bool | None = None
        self._watchers_last_checked_monotonic = 0.0
        self._watchers_cache_ttl_s = 30.0
        self._xdotool_available: bool | None = None
        self._ydotool_available: bool | None = None
        self._active_window_info: dict[str, object] | None = None
        self._active_window_last_checked_monotonic = 0.0
        self._active_window_cache_ttl_s = 1.0

    def _run(self, args: list[str], op: str, attempts: int | None = None) -> bool:
        attempts = attempts if attempts is not None else self.retry_attempts
        attempts = max(1, attempts)

        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(args, check=True, timeout=self.subprocess_timeout)
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

    def _xdotool_installed(self) -> bool:
        if self._xdotool_available is None:
            self._xdotool_available = shutil.which("xdotool") is not None
        return self._xdotool_available

    def _ydotool_installed(self) -> bool:
        if self._ydotool_available is None:
            self._ydotool_available = shutil.which("ydotool") is not None
        return self._ydotool_available

    def _detect_active_window(self) -> dict[str, object]:
        now = time.monotonic()
        if self._active_window_info is not None:
            age = now - self._active_window_last_checked_monotonic
            if age < self._active_window_cache_ttl_s:
                return self._active_window_info

        payload: dict[str, object] = {}
        try:
            result = subprocess.run(
                ["hyprctl", "activewindow", "-j"],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
            loaded = json.loads(result.stdout or "{}")
            if isinstance(loaded, dict):
                payload = loaded
        except Exception as e:
            log.debug("Could not inspect active window for injection backend: %s", e)

        self._active_window_info = payload
        self._active_window_last_checked_monotonic = now
        return payload

    def _active_window_is_xwayland(self) -> bool:
        return bool(self._detect_active_window().get("xwayland"))

    def _active_xdotool_window_id(self) -> str | None:
        window = self._detect_active_window()
        pid = window.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return None

        try:
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--pid", str(pid)],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
        except Exception as e:
            log.debug("Could not resolve X11 window id for pid %s: %s", pid, e)
            return None

        window_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return window_ids[-1] if window_ids else None

    def _prefer_xdotool(self) -> bool:
        return self._xdotool_installed() and self._active_window_is_xwayland()

    def _prefer_ydotool(self) -> bool:
        return self._ydotool_installed() and self._active_window_is_xwayland()

    @staticmethod
    def _backspace_args(count: int) -> list[str]:
        args = ["wtype"]
        for _ in range(count):
            args.extend(["-k", "BackSpace"])
        return args

    @staticmethod
    def _ydotool_backspace_args(count: int) -> list[str]:
        args = ["ydotool", "key", "-d", str(_YDOTOOL_KEY_DELAY_MS)]
        for _ in range(count):
            args.extend([f"{_KEY_BACKSPACE}:1", f"{_KEY_BACKSPACE}:0"])
        return args

    def _xdotool_key_args(
        self,
        key_sequence: str,
        window_id: str | None = None,
        repeat: int | None = None,
    ) -> list[str]:
        args = ["xdotool", "key", "--clearmodifiers", "--delay", "0"]
        if window_id:
            args.extend(["--window", window_id])
        if repeat and repeat > 1:
            args.extend(["--repeat", str(repeat), "--repeat-delay", "0"])
        args.append(key_sequence)
        return args

    def _xdotool_type_args(self, text: str, window_id: str | None = None) -> list[str]:
        args = ["xdotool", "type", "--clearmodifiers", "--delay", "0"]
        if window_id:
            args.extend(["--window", window_id])
        args.append(text)
        return args

    def _send_backspaces_via_xdotool(self, count: int, op: str) -> bool:
        if count <= 0:
            return True

        window_id = self._active_xdotool_window_id()
        remaining = count
        while remaining > 0:
            batch = min(remaining, _BACKSPACE_BATCH_SIZE)
            ok = self._run(
                self._xdotool_key_args("BackSpace", window_id, repeat=batch),
                op,
                attempts=1,
            )
            if not ok:
                return False
            remaining -= batch
        return True

    def _send_backspaces_via_ydotool(self, count: int, op: str) -> bool:
        if count <= 0:
            return True

        remaining = count
        while remaining > 0:
            batch = min(remaining, _BACKSPACE_BATCH_SIZE)
            ok = self._run(self._ydotool_backspace_args(batch), op)
            if not ok:
                return False
            remaining -= batch
        return True

    def _send_backspaces(self, count: int, op: str) -> bool:
        if count <= 0:
            return True

        prefer_xdotool = self._prefer_xdotool()
        prefer_ydotool = self._prefer_ydotool()
        if prefer_xdotool and self._send_backspaces_via_xdotool(
            count, "xdotool partial backspace"
        ):
            return True
        if prefer_ydotool and self._send_backspaces_via_ydotool(
            count, "ydotool partial backspace"
        ):
            return True

        remaining = count
        while remaining > 0:
            batch = min(remaining, _BACKSPACE_BATCH_SIZE)
            ok = self._run(self._backspace_args(batch), op)
            if not ok:
                if self._xdotool_installed() and self._active_window_is_xwayland() and not prefer_xdotool:
                    return self._send_backspaces_via_xdotool(
                        remaining, "xdotool partial backspace"
                    )
                if self._ydotool_installed() and not prefer_ydotool:
                    return self._send_backspaces_via_ydotool(
                        remaining, "ydotool partial backspace"
                    )
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
                timeout=self.subprocess_timeout,
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

        prefer_xdotool = self._prefer_xdotool()
        prefer_ydotool = self._prefer_ydotool()
        attempted_xdotool = False
        attempted_ydotool = False
        if prefer_xdotool:
            attempted_xdotool = True
            log.info("Using xdotool direct typing for focused XWayland window.")
            typed = self._run(
                self._xdotool_type_args(text, self._active_xdotool_window_id()),
                "xdotool direct type",
            )
            if typed:
                return True

        if prefer_ydotool:
            attempted_ydotool = True
            log.info("Using ydotool direct typing for focused XWayland window.")
            typed = self._run(
                [
                    "ydotool",
                    "type",
                    "--key-delay",
                    str(_YDOTOOL_KEY_DELAY_MS),
                    "--key-hold",
                    str(_YDOTOOL_HOLD_DELAY_MS),
                    "--",
                    text,
                ],
                "ydotool direct type",
            )
            if typed:
                return True

        typed = self._run(["wtype", "--", text], "wtype direct type")
        if typed:
            return typed

        if self._xdotool_installed() and self._active_window_is_xwayland() and not attempted_xdotool:
            attempted_xdotool = True
            log.info("Falling back to xdotool direct typing for focused XWayland window.")
            typed = self._run(
                self._xdotool_type_args(text, self._active_xdotool_window_id()),
                "xdotool direct type",
            )
            if typed:
                return True

        if attempted_ydotool or not self._ydotool_installed():
            return False

        return self._run(
            [
                "ydotool",
                "type",
                "--key-delay",
                str(_YDOTOOL_KEY_DELAY_MS),
                "--key-hold",
                str(_YDOTOOL_HOLD_DELAY_MS),
                "--",
                text,
            ],
            "ydotool direct type",
        )

    def _paste_via_clipboard(self, text: str) -> bool:
        if not text:
            return True

        copied = self._run(["wl-copy", "--", text], "wl-copy set")
        if not copied:
            return False

        if self.clipboard_settle_delay_s > 0:
            time.sleep(self.clipboard_settle_delay_s)

        prefer_xdotool = self._prefer_xdotool()
        prefer_ydotool = self._prefer_ydotool()
        attempted_xdotool = False
        attempted_ydotool = False
        if prefer_xdotool:
            attempted_xdotool = True
            log.info("Using xdotool Ctrl+V paste for focused XWayland window.")
            pasted = self._run(
                self._xdotool_key_args("ctrl+v", self._active_xdotool_window_id()),
                "xdotool ctrl+v",
            )
            if pasted:
                return True

        if prefer_ydotool:
            attempted_ydotool = True
            log.info("Using ydotool Ctrl+V paste for focused XWayland window.")
            pasted = self._run(
                [
                    "ydotool",
                    "key",
                    "-d",
                    str(_YDOTOOL_KEY_DELAY_MS),
                    f"{_KEY_LEFTCTRL}:1",
                    f"{_KEY_V}:1",
                    f"{_KEY_V}:0",
                    f"{_KEY_LEFTCTRL}:0",
                ],
                "ydotool ctrl+v",
            )
            if pasted:
                return True

        pasted = self._run(
            ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
            "wtype ctrl+v",
        )
        if pasted:
            return pasted

        if self._xdotool_installed() and self._active_window_is_xwayland() and not attempted_xdotool:
            attempted_xdotool = True
            log.info("Falling back to xdotool Ctrl+V paste for focused XWayland window.")
            pasted = self._run(
                self._xdotool_key_args("ctrl+v", self._active_xdotool_window_id()),
                "xdotool ctrl+v",
            )
            if pasted:
                return True

        if attempted_ydotool or not self._ydotool_installed():
            return False

        return self._run(
            [
                "ydotool",
                "key",
                "-d",
                str(_YDOTOOL_KEY_DELAY_MS),
                f"{_KEY_LEFTCTRL}:1",
                f"{_KEY_V}:1",
                f"{_KEY_V}:0",
                f"{_KEY_LEFTCTRL}:0",
            ],
            "ydotool ctrl+v",
        )

    def _capture_clipboard(self) -> tuple[bool, str]:
        """Return (had_content, content). Best effort only."""
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                check=True,
                timeout=self.subprocess_timeout,
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
        final_text = sanitize_final_injection_text(final_text)

        use_clipboard = True
        if self.final_injection_mode == "direct":
            use_clipboard = False
        elif self.final_injection_mode == "auto":
            # XWayland editors tend to behave better with clipboard paste than
            # synthetic per-character typing, even when clipboard watchers exist.
            prefer_xdotool = self._prefer_xdotool()
            if prefer_xdotool:
                log.info("Auto mode selected clipboard paste for focused XWayland window.")
            use_clipboard = prefer_xdotool or not self._detect_clipboard_watchers()

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
