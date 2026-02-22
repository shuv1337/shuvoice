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

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHERPA_MODEL_DIR = (
    ROOT / "build" / "asr-models" / "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
)

# Phrase cases from user-provided Sherpa GPU low-noise notes.
# Thresholds intentionally mirror expected real-world behavior:
# - easy phrases should remain near-perfect
# - known hard phrases (tongue twister + invoice numbers) allow lower similarity
PHRASE_CASES = [
    {
        "text": "The quick brown fox jumped over the lazy dog",
        "repeats": 4,
        "min_median_similarity": 0.85,
        "max_empty_ratio": 0.0,
    },
    {
        "text": "Please schedule the moonshine demo for Wednesday at three thirty",
        "repeats": 4,
        "min_median_similarity": 0.50,
        "max_empty_ratio": 0.0,
    },
    {
        "text": "The sixth sick sheik's sixth sheep's sick",
        "repeats": 4,
        "min_median_similarity": 0.55,
        "max_empty_ratio": 0.0,
    },
    {
        "text": "Invoice 4827 totals one hundred and fifty three dollars and twelve cents",
        "repeats": 7,
        "min_median_similarity": 0.60,
        "max_empty_ratio": 0.0,
    },
    {
        "text": "Hey, can you review the pull request? I left a couple comments on the config changes",
        "repeats": 4,
        "min_median_similarity": 0.80,
        "max_empty_ratio": 0.0,
    },
    {
        "text": "The function returns an empty string when the input buffer is less than three hundred and fifty milliseconds",
        "repeats": 4,
        "min_median_similarity": 0.85,
        "max_empty_ratio": 0.0,
    },
]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _enabled() -> bool:
    return _env("SHUVOICE_RUN_SHERPA_LOW_NOISE", "0") in {"1", "true", "TRUE", "yes"}


def _require_enabled():
    if not _enabled():
        pytest.skip("Set SHUVOICE_RUN_SHERPA_LOW_NOISE=1 to run Sherpa GPU low-noise regression")


def _roundtrip_python() -> str:
    override = _env("SHUVOICE_ROUNDTRIP_PYTHON", "")
    if override:
        return override

    return sys.executable


def test_sherpa_gpu_low_noise_phrase_regression(tmp_path: Path):
    _require_enabled()

    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        pytest.skip("espeak-ng/espeak is required for roundtrip fixture generation")

    model_dir = Path(_env("SHUVOICE_SHERPA_MODEL_DIR", str(DEFAULT_SHERPA_MODEL_DIR)))
    if not model_dir.exists():
        pytest.skip(f"Sherpa model dir not found: {model_dir}")

    sherpa_provider = _env("SHUVOICE_SHERPA_PROVIDER", "cuda")
    flush_chunks = int(_env("SHUVOICE_SHERPA_FLUSH_CHUNKS", "8"))
    timeout_sec = int(_env("SHUVOICE_SHERPA_TIMEOUT_SEC", "2400"))

    max_empty_ratio_total = float(_env("SHUVOICE_SHERPA_MAX_EMPTY_RATIO", "0.03"))
    min_median_similarity_total = float(_env("SHUVOICE_SHERPA_MIN_MEDIAN_SIM", "0.65"))

    phrase_lines: list[str] = []
    for case in PHRASE_CASES:
        phrase_lines.extend([case["text"]] * int(case["repeats"]))

    phrases_file = tmp_path / "sherpa-low-noise-phrases.txt"
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
        "sherpa",
        "--sherpa-model-dir",
        str(model_dir),
        "--sherpa-provider",
        sherpa_provider,
        "--flush-chunks",
        str(flush_chunks),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    assert proc.returncode == 0, (
        f"Sherpa low-noise roundtrip failed (exit={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )

    csv_path = output_dir / "roundtrip.csv"
    assert csv_path.exists(), "roundtrip.csv was not produced"

    rows: list[dict[str, str]] = []
    with csv_path.open() as f:
        rows.extend(csv.DictReader(f))

    assert len(rows) == len(phrase_lines), (
        f"Unexpected row count. expected={len(phrase_lines)} actual={len(rows)}"
    )

    empty_total = sum(1 for row in rows if not row["hypothesis"].strip())
    total_empty_ratio = empty_total / len(rows)

    similarities = [float(row["similarity"]) for row in rows]
    total_median_similarity = statistics.median(similarities)

    assert total_empty_ratio <= max_empty_ratio_total, (
        "Total empty hypothesis ratio exceeded threshold. "
        f"ratio={total_empty_ratio:.3f} threshold={max_empty_ratio_total:.3f}"
    )
    assert total_median_similarity >= min_median_similarity_total, (
        "Total median similarity below threshold. "
        f"median={total_median_similarity:.3f} threshold={min_median_similarity_total:.3f}"
    )

    rows_by_phrase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_phrase[row["reference"]].append(row)

    for case in PHRASE_CASES:
        phrase = str(case["text"])
        phrase_rows = rows_by_phrase[phrase]
        assert phrase_rows, f"No rows produced for phrase: {phrase!r}"

        phrase_empty = sum(1 for row in phrase_rows if not row["hypothesis"].strip())
        phrase_empty_ratio = phrase_empty / len(phrase_rows)
        phrase_median_similarity = statistics.median(
            float(row["similarity"]) for row in phrase_rows
        )

        max_empty_ratio = float(case["max_empty_ratio"])
        min_median_similarity = float(case["min_median_similarity"])

        assert phrase_empty_ratio <= max_empty_ratio, (
            "Per-phrase empty ratio exceeded threshold. "
            f"phrase={phrase!r} ratio={phrase_empty_ratio:.3f} threshold={max_empty_ratio:.3f}"
        )
        assert phrase_median_similarity >= min_median_similarity, (
            "Per-phrase median similarity below threshold. "
            f"phrase={phrase!r} median={phrase_median_similarity:.3f} "
            f"threshold={min_median_similarity:.3f}"
        )
