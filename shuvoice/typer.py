"""Text injection via wtype and wl-clipboard."""

import logging
import subprocess

log = logging.getLogger(__name__)


class StreamingTyper:
    """Injects text into the focused Wayland window.

    Uses wtype for keystroke simulation and wl-copy + Ctrl+V for
    clipboard-based paste of final transcriptions.
    """

    def __init__(self):
        self.last_partial_len = 0

    def update_partial(self, new_text: str):
        """Replace previous partial text with new partial text via backspace+retype."""
        if not new_text and self.last_partial_len == 0:
            return

        args = ["wtype"]
        for _ in range(self.last_partial_len):
            args.extend(["-k", "BackSpace"])
        if new_text:
            args.extend(["--", new_text])

        try:
            subprocess.run(args, check=True, timeout=5)
        except subprocess.SubprocessError as e:
            log.error("wtype partial update failed: %s", e)

        self.last_partial_len = len(new_text)

    def commit_final(self, final_text: str):
        """Erase partial text, then paste final text via clipboard."""
        # Erase any remaining partial text
        if self.last_partial_len > 0:
            args = ["wtype"]
            for _ in range(self.last_partial_len):
                args.extend(["-k", "BackSpace"])
            try:
                subprocess.run(args, check=True, timeout=5)
            except subprocess.SubprocessError as e:
                log.error("wtype backspace failed: %s", e)

        if final_text:
            try:
                # wl-copy -- handles text starting with dashes
                subprocess.run(
                    ["wl-copy", "--", final_text], check=True, timeout=5
                )
                # Fix #4: use -k v for explicit key press instead of bare v
                subprocess.run(
                    ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
                    check=True,
                    timeout=5,
                )
            except subprocess.SubprocessError as e:
                log.error("Clipboard paste failed: %s", e)

        self.last_partial_len = 0

    def reset(self):
        """Reset tracking state without sending any keystrokes."""
        self.last_partial_len = 0
