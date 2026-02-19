"""Push-to-talk hotkey handling via evdev.

Implements a hybrid tap/hold pattern:
  - Hold > threshold: push-to-talk (stop on release)
  - Tap < threshold: toggle recording on/off
"""

from __future__ import annotations

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


def find_keyboards(hotkey_code: int) -> list[str]:
    """Find likely keyboard devices that can emit the target hotkey code."""
    matches: list[str] = []

    for path in evdev.list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities()
        if ecodes.EV_KEY not in caps:
            continue

        keys = caps[ecodes.EV_KEY]
        # Heuristic: real keyboard-like device with common alpha/enter keys.
        if ecodes.KEY_A not in keys or ecodes.KEY_ENTER not in keys:
            continue

        if hotkey_code not in keys:
            continue

        matches.append(dev.path)
        log.debug("Keyboard candidate: %s (%s)", dev.name, dev.path)

    if not matches:
        raise RuntimeError(
            "No keyboard device found for configured hotkey. "
            "Ensure user is in 'input' group: sudo usermod -aG input $USER"
        )

    return matches


class HotkeyListener:
    def __init__(
        self,
        hotkey_name: str = "KEY_RIGHTCTRL",
        hold_threshold_ms: int = 300,
        device_path: str | None = None,
    ):
        if not hasattr(ecodes, hotkey_name):
            raise ValueError(
                f"Unknown hotkey '{hotkey_name}'. "
                "Use an evdev key name like KEY_RIGHTCTRL or KEY_F9."
            )

        self.hotkey_code: int = getattr(ecodes, hotkey_name)
        self.hold_threshold_s = hold_threshold_ms / 1000.0
        self.device_path = device_path

        self._on_start: Callable | None = None
        self._on_stop: Callable | None = None
        self._state = _IDLE
        self._press_time = 0.0

    def set_callbacks(self, on_start: Callable, on_stop: Callable):
        """Set recording start/stop callbacks. Called from the evdev thread."""
        self._on_start = on_start
        self._on_stop = on_stop

    async def _listen_device(self, device: InputDevice):
        async for event in device.async_read_loop():
            if event.type != ecodes.EV_KEY or event.code != self.hotkey_code:
                continue

            if event.value == 1:  # Key press
                self._on_key_down()
            elif event.value == 0:  # Key release
                self._on_key_up()
            # event.value == 2 (auto-repeat) is ignored

    async def run(self):
        """Run the hotkey event loop. Call from a dedicated thread."""
        if self.device_path:
            paths = [self.device_path]
        else:
            paths = find_keyboards(self.hotkey_code)

        devices = [InputDevice(path) for path in paths]
        for dev in devices:
            log.info(
                "Listening for %s on %s (%s)",
                ecodes.KEY.get(self.hotkey_code, self.hotkey_code),
                dev.name,
                dev.path,
            )

        await asyncio.gather(*(self._listen_device(device) for device in devices))

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
