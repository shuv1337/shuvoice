from __future__ import annotations

from unittest.mock import patch

from shuvoice.config import Config
from shuvoice.waybar import _action_menu, _perform_action, build_waybar_payload, config_info_lines


def test_build_waybar_payload_recording_state():
    payload = build_waybar_payload("recording")

    assert payload["text"] == "\uf130"
    assert payload["alt"] == "recording"
    assert payload["class"] == "recording"
    assert "ShuVoice: Recording" in payload["tooltip"]


def test_build_waybar_payload_stopped_state():
    payload = build_waybar_payload("stopped")

    assert payload["text"] == "\uf131"
    assert payload["alt"] == "stopped"
    assert payload["class"] == "stopped"
    assert "ShuVoice: Stopped" in payload["tooltip"]


def test_build_waybar_payload_error_reason_class_is_sanitized():
    payload = build_waybar_payload("error:asr_thread_dead")

    assert payload["text"] == "\uf071"
    assert payload["alt"] == "error"
    assert payload["class"] == "error"
    assert "Reason: asr_thread_dead" in payload["tooltip"]


def test_build_waybar_payload_includes_service_and_control_details_for_error_states():
    payload = build_waybar_payload(
        "error:service_failed",
        service_state="failed",
        control_error="Control socket not found",
        action_error="systemctl restart failed",
    )

    assert payload["class"] == "error"
    assert "Service: failed" in payload["tooltip"]
    assert "Control: Control socket not found" in payload["tooltip"]
    assert "Action: systemctl restart failed" in payload["tooltip"]


# -- config_lines integration -------------------------------------------------


def test_build_waybar_payload_includes_config_lines_in_tooltip():
    info = ["Backend:  NeMo (NVIDIA)", "Model:    nemotron", "Device:   GPU (CUDA)"]
    payload = build_waybar_payload("idle", config_lines=info)

    tooltip = payload["tooltip"]
    assert "Backend:  NeMo (NVIDIA)" in tooltip
    assert "Model:    nemotron" in tooltip
    assert "Device:   GPU (CUDA)" in tooltip
    # Config info should appear before the click actions
    assert tooltip.index("Backend:") < tooltip.index("Left click:")


def test_build_waybar_payload_without_config_lines_still_works():
    payload = build_waybar_payload("idle")
    assert "Left click:" in payload["tooltip"]
    assert "Right click: open action menu" in payload["tooltip"]


def test_perform_action_launch_wizard_calls_detached_launcher():
    with patch("shuvoice.waybar._launch_wizard_detached") as launch_wizard:
        _perform_action("launch-wizard", Config(), "shuvoice.service")

    launch_wizard.assert_called_once_with()


def test_perform_action_menu_calls_menu_handler():
    with patch("shuvoice.waybar._action_menu") as action_menu:
        _perform_action("menu", Config(), "shuvoice.service")

    action_menu.assert_called_once()


def test_action_menu_dispatches_selected_command():
    config = Config()
    with patch("shuvoice.waybar._query_runtime_state", return_value=("idle", None, None)), patch(
        "shuvoice.waybar._service_active_state", return_value="active"
    ), patch("shuvoice.waybar._prompt_menu_choice", return_value="Relaunch setup wizard"), patch(
        "shuvoice.waybar._perform_action"
    ) as perform_action:
        _action_menu(config, "shuvoice.service")

    perform_action.assert_called_once_with("launch-wizard", config, "shuvoice.service")


# -- config_info_lines ---------------------------------------------------------


def test_config_info_lines_nemo():
    cfg = Config(asr_backend="nemo", model_name="nvidia/nemotron-speech-streaming-en-0.6b", device="cuda")
    lines = config_info_lines(cfg)

    assert any("NeMo" in line for line in lines)
    assert any("nemotron-speech-streaming-en-0.6b" in line for line in lines)
    assert any("GPU (CUDA)" in line for line in lines)


def test_config_info_lines_nemo_cpu():
    cfg = Config(asr_backend="nemo", device="cpu")
    lines = config_info_lines(cfg)

    assert any("CPU" in line for line in lines)


def test_config_info_lines_sherpa_default_model():
    cfg = Config(asr_backend="sherpa", sherpa_provider="cpu")
    lines = config_info_lines(cfg)

    assert any("Sherpa-ONNX" in line for line in lines)
    assert any("auto-download" in line for line in lines)
    assert any("CPU" in line for line in lines)


def test_config_info_lines_sherpa_custom_model():
    cfg = Config(asr_backend="sherpa", sherpa_model_dir="/opt/models/my-zipformer")
    lines = config_info_lines(cfg)

    assert any("my-zipformer" in line for line in lines)


def test_config_info_lines_moonshine():
    cfg = Config(asr_backend="moonshine", moonshine_model_name="moonshine/tiny", moonshine_provider="cpu")
    lines = config_info_lines(cfg)

    assert any("Moonshine-ONNX" in line for line in lines)
    assert any("moonshine/tiny" in line for line in lines)
    assert any("CPU" in line for line in lines)
