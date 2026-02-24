#!/usr/bin/env python3
"""Validate config key drift between code and docs/examples."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from shuvoice.config import CONFIG_SECTION_FIELDS, Config

ROOT = Path(__file__).resolve().parents[1]

KNOWN_SECTIONS = set(CONFIG_SECTION_FIELDS)
KNOWN_FIELDS = Config.config_field_names()
KNOWN_TOP_LEVEL = {"config_version"}


def _flatten_tables(data: dict) -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        if key in KNOWN_TOP_LEVEL:
            keys.add(key)
            continue
        if isinstance(value, dict):
            keys.update(value.keys())
        else:
            keys.add(key)
    return keys


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _extract_toml_blocks(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    pattern = re.compile(r"```toml\n(.*?)```", re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in pattern.findall(text)]


def _parse_toml_snippet(snippet: str) -> dict | None:
    try:
        return tomllib.loads(snippet)
    except Exception:
        return None


def _validate_examples_config() -> list[str]:
    errors: list[str] = []
    path = ROOT / "examples" / "config.toml"
    data = _load_toml(path)

    if "config_version" not in data:
        errors.append("examples/config.toml is missing top-level `config_version`.")

    unknown_sections = sorted(
        key for key, value in data.items() if isinstance(value, dict) and key not in KNOWN_SECTIONS
    )
    if unknown_sections:
        errors.append(
            "examples/config.toml contains unknown sections: " + ", ".join(unknown_sections)
        )

    unknown_keys = sorted(_flatten_tables(data) - (KNOWN_FIELDS | KNOWN_TOP_LEVEL))
    if unknown_keys:
        errors.append(
            "examples/config.toml contains unknown config keys: " + ", ".join(unknown_keys)
        )

    return errors


def _validate_markdown_toml_keys(markdown_path: Path) -> list[str]:
    errors: list[str] = []
    all_keys: set[str] = set()

    for snippet in _extract_toml_blocks(markdown_path):
        parsed = _parse_toml_snippet(snippet)
        if parsed is None:
            continue
        all_keys.update(_flatten_tables(parsed))

    unknown = sorted(all_keys - (KNOWN_FIELDS | KNOWN_TOP_LEVEL))
    if unknown:
        errors.append(f"{markdown_path.name} references unknown config keys: " + ", ".join(unknown))

    return errors


def main() -> int:
    errors: list[str] = []

    errors.extend(_validate_examples_config())
    for doc_name in ("README.md", "AGENTS.md"):
        errors.extend(_validate_markdown_toml_keys(ROOT / doc_name))

    if errors:
        print("Config/docs drift detected:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Config/docs drift check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
