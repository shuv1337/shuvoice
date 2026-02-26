from __future__ import annotations

from shuvoice import __main__ as entry


def test_main_delegates_to_cli_main(monkeypatch):
    monkeypatch.setattr(entry, "_cli_main", lambda argv=None: 42)

    assert entry.main(["--help"]) == 42
