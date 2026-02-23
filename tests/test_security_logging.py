from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shuvoice.asr_moonshine import MoonshineBackend
from shuvoice.utterance_state import _UtteranceState

pytest.importorskip("gi")
from shuvoice.app import ShuVoiceApp


def _state_with_chunk(*, last_text: str = "") -> _UtteranceState:
    state = _UtteranceState(last_text=last_text)
    state.reset(rms_threshold=0.01)
    state.last_text = last_text
    state.add_chunk(np.zeros(1000, dtype=np.float32))
    return state


@pytest.fixture
def app_stub() -> SimpleNamespace:
    return SimpleNamespace(
        asr=SimpleNamespace(native_chunk_samples=1000, wants_raw_audio=True, debug_step_num=1),
        audio=SimpleNamespace(queue=SimpleNamespace(qsize=lambda: 0)),
        _process_chunk_safe=Mock(return_value="Sensitive Password"),
        _recover_asr_after_failure=Mock(),
        _render_transcript_text=lambda text: text,
        overlay=None,
        config=SimpleNamespace(output_mode="final_only", use_clipboard_for_final=True),
        typer=SimpleNamespace(update_partial=Mock(), commit_final=Mock(), reset=Mock()),
    )


def test_transcribe_logs_length_not_raw_text(app_stub: SimpleNamespace, caplog):
    caplog.set_level(logging.DEBUG, logger="shuvoice.app")
    state = _state_with_chunk()

    ShuVoiceApp._transcribe_native_chunk(app_stub, state, "test context")

    messages = [record.getMessage() for record in caplog.records]
    assert all("Sensitive Password" not in message for message in messages)
    assert any("raw_text_len=" in message for message in messages)


def test_commit_logs_length_not_final_text(caplog):
    caplog.set_level(logging.INFO, logger="shuvoice.app")

    app = SimpleNamespace(
        _render_transcript_text=lambda text: text,
        overlay=None,
        typer=SimpleNamespace(commit_final=Mock(), update_partial=Mock(), reset=Mock()),
        config=SimpleNamespace(use_clipboard_for_final=True),
    )
    state = _UtteranceState(last_text="Final Sensitive Text")

    ShuVoiceApp._commit_utterance(app, state)

    messages = [record.getMessage() for record in caplog.records]
    assert all("Final Sensitive Text" not in message for message in messages)
    assert any("Final: len=" in message for message in messages)


def test_moonshine_repetition_logs_pattern_size_not_words(caplog):
    caplog.set_level(logging.DEBUG, logger="shuvoice.asr_moonshine")

    secret = "SecretPassword"
    text = (f"{secret} foo " * 5).strip()

    MoonshineBackend._guard_repetition(text, audio_seconds=5.0)

    messages = [record.getMessage() for record in caplog.records]
    assert all(secret not in message for message in messages)
    assert any("word pattern repeated" in message for message in messages)
