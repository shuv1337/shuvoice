
import logging
import subprocess
from unittest.mock import patch

from shuvoice.typer import StreamingTyper


def test_typer_does_not_log_sensitive_text(caplog):
    """Verify that sensitive text is not logged when subprocess fails."""
    caplog.set_level(logging.ERROR)

    typer = StreamingTyper()
    sensitive_text = "SECRET_PASSWORD"

    # Mock subprocess.run to raise CalledProcessError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["wtype", "--", sensitive_text]
        )

        # This should trigger the error logging
        typer._type_direct(sensitive_text)

        # Check logs
        for record in caplog.records:
            assert sensitive_text not in record.message, f"Sensitive text found in log: {record.message}"
            assert "wtype direct type failed" in record.message
