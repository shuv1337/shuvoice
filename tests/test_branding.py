from __future__ import annotations

from pathlib import Path

from shuvoice.branding import find_logo, logo_candidates


def test_logo_candidates_include_packaged_doc_paths() -> None:
    candidates = {str(path) for path in logo_candidates()}

    assert any(
        candidate.startswith("/usr/share/doc/shuvoice/docs/assets/branding/")
        for candidate in candidates
    )
    assert any(
        candidate.startswith("/usr/share/doc/shuvoice-git/docs/assets/branding/")
        for candidate in candidates
    )


def test_find_logo_returns_first_existing_candidate(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    second.write_bytes(b"\x89PNG")

    monkeypatch.setattr(
        "shuvoice.branding.logo_candidates",
        lambda: (first, second),
    )

    assert find_logo() == second
