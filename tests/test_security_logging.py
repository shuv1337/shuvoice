import logging
from unittest.mock import MagicMock

import pytest

from shuvoice.app import ShuVoiceApp
from shuvoice.config import Config
from shuvoice.utterance_state import _UtteranceState


@pytest.fixture
def app():
    config = Config()
    # Mock dependencies to avoid side effects
    with pytest.MonkeyPatch.context() as m:
        m.setattr("shuvoice.app.create_backend", MagicMock())
        m.setattr("shuvoice.app.AudioCapture", MagicMock())
        m.setattr("shuvoice.app.StreamingTyper", MagicMock())
        m.setattr("shuvoice.app.ControlServer", MagicMock())
        m.setattr("shuvoice.app.HotkeyListener", MagicMock())

        # Instantiate app
        app = ShuVoiceApp(config)
        # Mock ASR backend
        app.asr = MagicMock()
        app.asr.native_chunk_samples = 1000
        app.asr.wants_raw_audio = False
        app.asr.process_chunk.return_value = "Sensitive Password"

        # Mock audio queue
        app.audio = MagicMock()
        app.audio.queue.qsize.return_value = 0

        return app


def test_no_sensitive_data_in_debug_logs(app, caplog):
    """Verify that transcribed text is not logged in DEBUG logs."""
    caplog.set_level(logging.DEBUG)

    state = _UtteranceState()
    state.reset(rms_threshold=0.01)
    # Mock some audio data
    import numpy as np

    chunk = np.zeros(1000, dtype=np.float32)
    state.add_chunk(chunk)

    # Run _transcribe_native_chunk
    # We need to simulate enough data for a chunk

    # We mock _process_chunk_safe to return sensitive text
    app._process_chunk_safe = MagicMock(return_value="Sensitive Password")

    # We call _transcribe_native_chunk
    app._transcribe_native_chunk(state, "test context")

    # Check logs
    for record in caplog.records:
        if "Sensitive Password" in record.message:
            pytest.fail(f"Sensitive data found in log: {record.message}")

    # Also check if length is logged instead
    for record in caplog.records:
        if "raw_text_len=" in record.message or "text_len" in record.message:
            pass
        # Or checking if it mentions length
        if "raw_text=" in record.message:
            # If it says raw_text=..., it might be the old format.
            # We want to ensure it DOESN'T verify sensitive text, which we did above.
            pass


def test_no_sensitive_data_in_transcript_update_logs(app, caplog):
    caplog.set_level(logging.DEBUG)
    state = _UtteranceState()
    state.last_text = "Old Text"

    app._process_chunk_safe = MagicMock(return_value="New Sensitive Text")

    # Simulate update
    import numpy as np

    chunk = np.zeros(1000, dtype=np.float32)
    state.add_chunk(chunk)

    app._transcribe_native_chunk(state, "test context")

    for record in caplog.records:
        if "New Sensitive Text" in record.message:
            pytest.fail(f"Sensitive data found in log: {record.message}")


def test_no_sensitive_data_in_final_logs(app, caplog):
    caplog.set_level(logging.INFO)
    state = _UtteranceState()
    state.last_text = "Final Sensitive Text"

    # Run _commit_utterance
    app._commit_utterance(state)

    for record in caplog.records:
        if "Final Sensitive Text" in record.message:
            pytest.fail(f"Sensitive data found in log: {record.message}")
