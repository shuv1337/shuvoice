from __future__ import annotations

import pytest

from shuvoice.config import Config


def test_overlay_font_size_validation():
    # Should be positive integer
    with pytest.raises(ValueError, match="font_size"):
        Config(font_size=0)

    with pytest.raises(ValueError, match="font_size"):
        Config(font_size=-1)

    # Strings should be rejected (or we can cast them, but better to reject if it's injection vector)
    # The fix will likely cast to int, so "22; ..." would fail int() conversion.
    with pytest.raises(ValueError):
        Config(font_size="22; background: red")


def test_overlay_bg_opacity_validation():
    # Should be float between 0.0 and 1.0
    with pytest.raises(ValueError, match="bg_opacity"):
        Config(bg_opacity=-0.1)

    with pytest.raises(ValueError, match="bg_opacity"):
        Config(bg_opacity=1.1)

    with pytest.raises(ValueError):
        Config(bg_opacity="0.5; color: red")


def test_overlay_border_radius_validation():
    # Should be non-negative integer
    with pytest.raises(ValueError, match="border_radius"):
        Config(border_radius=-1)

    with pytest.raises(ValueError):
        Config(border_radius="10; padding: 10px")


def test_overlay_bottom_margin_validation():
    # Should be non-negative integer
    with pytest.raises(ValueError, match="bottom_margin"):
        Config(bottom_margin=-1)

    with pytest.raises(ValueError):
        Config(bottom_margin="10; margin: 0")


def test_input_gain_validation():
    # Should be float > 0
    with pytest.raises(ValueError, match="input_gain"):
        Config(input_gain=0)

    with pytest.raises(ValueError, match="input_gain"):
        Config(input_gain=-1.0)


def test_sample_rate_validation():
    with pytest.raises(ValueError, match="sample_rate"):
        Config(sample_rate=0)

    with pytest.raises(ValueError, match="sample_rate"):
        Config(sample_rate=-16000)


def test_typing_retry_validation():
    with pytest.raises(ValueError, match="typing_retry_attempts"):
        Config(typing_retry_attempts=-1)

    with pytest.raises(ValueError, match="typing_retry_delay_ms"):
        Config(typing_retry_delay_ms=-1)
