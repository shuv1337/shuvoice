from __future__ import annotations

from unittest.mock import patch

from shuvoice.config import Config
from shuvoice.waybar import _toggle_debug_overlay, config_info_lines


def test_toggle_debug_overlay_enables_mode():
    cfg = Config(overlay_debug_mode=False)

    with (
        patch("shuvoice.cli.commands.config.config_set", return_value=0) as config_set,
        patch("shuvoice.waybar._service_action") as service_action,
        patch("shuvoice.waybar._wait_for_control_socket", return_value=True) as wait_for_socket,
    ):
        _toggle_debug_overlay(cfg, "shuvoice.service")

    config_set.assert_called_once_with("overlay_debug_mode", "true")
    service_action.assert_called_once_with("shuvoice.service", "restart")
    wait_for_socket.assert_called_once_with(cfg)
    assert cfg.overlay_debug_mode is True


def test_toggle_debug_overlay_disables_mode():
    cfg = Config(overlay_debug_mode=True)

    with (
        patch("shuvoice.cli.commands.config.config_set", return_value=0) as config_set,
        patch("shuvoice.waybar._service_action") as service_action,
        patch("shuvoice.waybar._wait_for_control_socket", return_value=True) as wait_for_socket,
    ):
        _toggle_debug_overlay(cfg, "shuvoice.service")

    config_set.assert_called_once_with("overlay_debug_mode", "false")
    service_action.assert_called_once_with("shuvoice.service", "restart")
    wait_for_socket.assert_called_once_with(cfg)
    assert cfg.overlay_debug_mode is False


def test_toggle_debug_overlay_raises_on_write_failure():
    cfg = Config(overlay_debug_mode=False)

    with patch("shuvoice.cli.commands.config.config_set", return_value=1):
        try:
            _toggle_debug_overlay(cfg, "shuvoice.service")
        except RuntimeError as exc:
            assert "failed to set overlay_debug_mode=true" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_toggle_debug_overlay_raises_when_service_restart_does_not_return():
    cfg = Config(overlay_debug_mode=False)

    with (
        patch("shuvoice.cli.commands.config.config_set", return_value=0),
        patch("shuvoice.waybar._service_action") as service_action,
        patch("shuvoice.waybar._wait_for_control_socket", return_value=False),
    ):
        try:
            _toggle_debug_overlay(cfg, "shuvoice.service")
        except RuntimeError as exc:
            assert "control socket not ready after restarting service" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    service_action.assert_called_once_with("shuvoice.service", "restart")


def test_waybar_config_info_lines_include_debug_overlay_state():
    enabled_lines = config_info_lines(Config(overlay_debug_mode=True))
    disabled_lines = config_info_lines(Config(overlay_debug_mode=False))

    assert "Debug:    Overlay on" in enabled_lines
    assert "Debug:    Overlay off" in disabled_lines
