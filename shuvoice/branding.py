"""Branding asset location helpers."""

from __future__ import annotations

import os
from pathlib import Path

# Order matters: runtime UI (splash/wizard) should prefer badge-style artwork
# without a full rectangular backdrop, then fall back to lockups.
_LOGO_FILENAMES = (
    "shuvoice-variant-dark-badge.png",
    "shuvoice_variant_dark_lockup_alt.png",
    "shuvoice-variant-dark-lockup.png",
    "shuvoice-variant-light-lockup.png",
)


def _branding_directories() -> tuple[Path, ...]:
    directories: list[Path] = []

    custom_dir = os.environ.get("SHUVOICE_BRANDING_DIR")
    if custom_dir:
        directories.append(Path(custom_dir).expanduser())

    # Repo checkout path (development/install-from-source workflows).
    repo_root = Path(__file__).resolve().parent.parent
    directories.append(repo_root / "docs" / "assets" / "branding")

    # Packaged doc paths (Arch/AUR and future non--git package names).
    directories.append(Path("/usr/share/doc/shuvoice/docs/assets/branding"))
    directories.append(Path("/usr/share/doc/shuvoice-git/docs/assets/branding"))

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in directories:
        resolved = directory.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(directory)

    return tuple(unique)


def logo_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for directory in _branding_directories():
        for filename in _LOGO_FILENAMES:
            candidates.append(directory / filename)
    return tuple(candidates)


def find_logo() -> Path | None:
    for candidate in logo_candidates():
        if candidate.is_file():
            return candidate
    return None
