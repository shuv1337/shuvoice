from __future__ import annotations

from shuvoice.postprocess import capitalize_first


def test_capitalize_first_basic_cases():
    assert capitalize_first("hello world") == "Hello world"
    assert capitalize_first(" already spaced") == " Already spaced"
    assert capitalize_first("123abc") == "123Abc"


def test_capitalize_first_noop_cases():
    assert capitalize_first("") == ""
    assert capitalize_first("12345") == "12345"
