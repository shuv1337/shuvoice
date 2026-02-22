from __future__ import annotations

from shuvoice.postprocess import apply_text_mappings, capitalize_first


def test_capitalize_first_basic_cases():
    assert capitalize_first("hello world") == "Hello world"
    assert capitalize_first(" already spaced") == " Already spaced"
    assert capitalize_first("123abc") == "123Abc"


def test_capitalize_first_noop_cases():
    assert capitalize_first("") == ""
    assert capitalize_first("12345") == "12345"


def test_apply_text_mappings_words_and_phrases():
    mappings = {
        "shove voice": "ShuVoice",
        "speech to text": "speech-to-text",
        "hyper land": "Hyprland",
    }

    text = "shove voice, the real-time speech to text overlay for hyper land"

    assert (
        apply_text_mappings(text, mappings)
        == "ShuVoice, the real-time speech-to-text overlay for Hyprland"
    )


def test_apply_text_mappings_respects_word_boundaries():
    mappings = {"land": "terrain"}

    assert apply_text_mappings("hyperland land", mappings) == "hyperland terrain"
