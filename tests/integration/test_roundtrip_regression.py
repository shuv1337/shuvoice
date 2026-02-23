from __future__ import annotations

import csv
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

MANUAL_REGRESSION_PHRASES = [
    "The quick brown fox jumped over the lazy dog",
    "Please schedule the moonshine demo for Wednesday at three thirty",
]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _enabled() -> bool:
    return _env("SHUVOICE_RUN_ROUNDTRIP", "0") in {"1", "true", "TRUE", "yes"}


def _require_roundtrip_enabled():
    if not _enabled():
        pytest.skip("Set SHUVOICE_RUN_ROUNDTRIP=1 to run backend roundtrip regression tests")


def _roundtrip_python() -> str:
    override = _env("SHUVOICE_ROUNDTRIP_PYTHON", "")
    if override:
        return override

    return sys.executable


@pytest.mark.gpu
def test_manual_phrase_roundtrip_regression(tmp_path: Path):
    _require_roundtrip_enabled()

    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        pytest.skip("espeak-ng/espeak is required for roundtrip fixture generation")

    repeats = int(_env("SHUVOICE_ROUNDTRIP_REPEATS", "5"))
    backend = _env("SHUVOICE_ROUNDTRIP_BACKEND", "nemo")
    flush_chunks = int(_env("SHUVOICE_ROUNDTRIP_FLUSH_CHUNKS", "5"))
    timeout_sec = int(_env("SHUVOICE_ROUNDTRIP_TIMEOUT_SEC", "1800"))

    max_empty_ratio_total = float(_env("SHUVOICE_ROUNDTRIP_MAX_EMPTY_RATIO", "0.10"))
    max_empty_ratio_per_phrase = float(
        _env("SHUVOICE_ROUNDTRIP_MAX_EMPTY_RATIO_PER_PHRASE", "0.20")
    )
    min_median_similarity_total = float(_env("SHUVOICE_ROUNDTRIP_MIN_MEDIAN_SIM", "0.90"))
    min_median_similarity_per_phrase = float(
        _env("SHUVOICE_ROUNDTRIP_MIN_MEDIAN_SIM_PER_PHRASE", "0.88")
    )

    phrase_lines: list[str] = []
    for phrase in MANUAL_REGRESSION_PHRASES:
        phrase_lines.extend([phrase] * repeats)

    phrases_file = tmp_path / "manual-regression-phrases.txt"
    phrases_file.write_text("\n".join(phrase_lines) + "\n")

    output_dir = tmp_path / "roundtrip-output"
    cmd = [
        _roundtrip_python(),
        "scripts/tts_roundtrip.py",
        "--phrases-file",
        str(phrases_file),
        "--output-dir",
        str(output_dir),
        "--asr-backend",
        backend,
        "--flush-chunks",
        str(flush_chunks),
    ]

    if backend == "nemo":
        nemo_device = _env("SHUVOICE_ROUNDTRIP_DEVICE", "cuda")
        cmd.extend(["--device", nemo_device])
    elif backend == "sherpa":
        sherpa_model_dir = _env("SHUVOICE_SHERPA_MODEL_DIR", "")
        if not sherpa_model_dir:
            pytest.skip(
                "SHUVOICE_SHERPA_MODEL_DIR is required when SHUVOICE_ROUNDTRIP_BACKEND=sherpa"
            )
        cmd.extend(["--sherpa-model-dir", sherpa_model_dir])
        cmd.extend(["--sherpa-provider", _env("SHUVOICE_SHERPA_PROVIDER", "cpu")])
    elif backend == "moonshine":
        moonshine_model_name = _env("SHUVOICE_MOONSHINE_MODEL_NAME", "moonshine/base")
        cmd.extend(["--moonshine-model-name", moonshine_model_name])

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    assert proc.returncode == 0, (
        f"Roundtrip harness failed (exit={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )

    csv_path = output_dir / "roundtrip.csv"
    assert csv_path.exists(), "roundtrip.csv was not produced"

    rows: list[dict[str, str]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    assert len(rows) == len(phrase_lines), (
        f"Unexpected row count. expected={len(phrase_lines)} actual={len(rows)}"
    )

    empty_total = sum(1 for row in rows if not row["hypothesis"].strip())
    total_empty_ratio = empty_total / len(rows)

    total_similarity = [float(row["similarity"]) for row in rows]
    total_median_similarity = statistics.median(total_similarity)

    assert total_empty_ratio <= max_empty_ratio_total, (
        "Roundtrip empty hypothesis ratio exceeded threshold. "
        f"ratio={total_empty_ratio:.3f} threshold={max_empty_ratio_total:.3f}"
    )
    assert total_median_similarity >= min_median_similarity_total, (
        "Roundtrip median similarity below threshold. "
        f"median={total_median_similarity:.3f} threshold={min_median_similarity_total:.3f}"
    )

    rows_by_phrase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_phrase[row["reference"]].append(row)

    for phrase in MANUAL_REGRESSION_PHRASES:
        phrase_rows = rows_by_phrase[phrase]
        assert phrase_rows, f"No rows produced for phrase: {phrase!r}"

        phrase_empty = sum(1 for row in phrase_rows if not row["hypothesis"].strip())
        phrase_empty_ratio = phrase_empty / len(phrase_rows)
        phrase_median_similarity = statistics.median(
            float(row["similarity"]) for row in phrase_rows
        )

        assert phrase_empty_ratio <= max_empty_ratio_per_phrase, (
            "Per-phrase empty hypothesis ratio exceeded threshold. "
            f"phrase={phrase!r} ratio={phrase_empty_ratio:.3f} "
            f"threshold={max_empty_ratio_per_phrase:.3f}"
        )
        assert phrase_median_similarity >= min_median_similarity_per_phrase, (
            "Per-phrase median similarity below threshold. "
            f"phrase={phrase!r} median={phrase_median_similarity:.3f} "
            f"threshold={min_median_similarity_per_phrase:.3f}"
        )
