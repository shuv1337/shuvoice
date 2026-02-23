from __future__ import annotations

from shuvoice.waybar import build_waybar_payload


def test_build_waybar_payload_recording_state():
    payload = build_waybar_payload("recording")

    assert payload["text"] == ""
    assert payload["alt"] == "recording"
    assert payload["class"] == "recording"
    assert "ShuVoice: Recording" in payload["tooltip"]


def test_build_waybar_payload_stopped_state():
    payload = build_waybar_payload("stopped")

    assert payload["text"] == ""
    assert payload["alt"] == "stopped"
    assert payload["class"] == "stopped"
    assert "ShuVoice: Stopped" in payload["tooltip"]


def test_build_waybar_payload_error_reason_class_is_sanitized():
    payload = build_waybar_payload("error:asr_thread_dead")

    assert payload["text"] == ""
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
