"""Push-to-talk hotkey handling via evdev.

Implements a hybrid tap/hold pattern:
  - Hold > threshold: push-to-talk (stop on release)
  - Tap < threshold: toggle recording on/off
"""

import asyncio
import logging
import time
from typing import Callable

import evdev
from evdev import InputDevice, ecodes

log = logging.getLogger(__name__)

# State machine
_IDLE = "idle"
_HELD = "held"  # Key is physically down, recording just started
_TOGGLED = "toggled"  # Tap-toggled on, waiting for next tap to stop
_STOPPING = "stopping"  # Key down in toggled state, waiting for release


def find_keyboard() -> str:
    """Find the first real keyboard device."""
    for path in evdev.list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities()
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            if ecodes.KEY_A in keys and ecodes.KEY_ENTER in keys:
                log.info("Found keyboard: %s (%s)", dev.name, dev.path)
                return dev.path
    raise RuntimeError(
        "No keyboard found. Ensure user is in 'input' group: "
        "sudo usermod -aG input $USER"
    )


class HotkeyListener:
    def __init__(self, hotkey_name: str = "KEY_RIGHTCTRL", hold_threshold_ms: int = 300):
        self.hotkey_code: int = getattr(ecodes, hotkey_name)
        self.hold_threshold_s = hold_threshold_ms / 1000.0
        self._on_start: Callable | None = None
        self._on_stop: Callable | None = None
        self._state = _IDLE
        self._press_time = 0.0

    def set_callbacks(self, on_start: Callable, on_stop: Callable):
        """Set recording start/stop callbacks. Called from the evdev thread."""
        self._on_start = on_start
        self._on_stop = on_stop

    async def run(self):
        """Run the hotkey event loop. Call from a dedicated thread."""
        device_path = find_keyboard()
        device = InputDevice(device_path)
        log.info(
            "Listening for %s on %s",
            ecodes.KEY.get(self.hotkey_code, self.hotkey_code),
            device.name,
        )

        async for event in device.async_read_loop():
            if event.type != ecodes.EV_KEY or event.code != self.hotkey_code:
                continue

            if event.value == 1:  # Key press
                self._on_key_down()
            elif event.value == 0:  # Key release
                self._on_key_up()

    def _on_key_down(self):
        if self._state == _IDLE:
            self._press_time = time.monotonic()
            self._state = _HELD
            if self._on_start:
                self._on_start()
        elif self._state == _TOGGLED:
            # Second tap: stop recording
            self._state = _STOPPING
            if self._on_stop:
                self._on_stop()

    def _on_key_up(self):
        if self._state == _HELD:
            held = time.monotonic() - self._press_time
            if held >= self.hold_threshold_s:
                # Push-to-talk: stop on release
                self._state = _IDLE
                if self._on_stop:
                    self._on_stop()
            else:
                # Tap: toggle on, stays recording
                self._state = _TOGGLED
        elif self._state == _STOPPING:
            # Release after toggle-off press
            self._state = _IDLE
