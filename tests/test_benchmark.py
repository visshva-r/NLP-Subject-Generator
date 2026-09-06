"""CLI benchmark smoke with a fake AESLC frame (no network, no GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_run_benchmark():
    path = ROOT / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_aeslc(monkeypatch):
    frame = pd.DataFrame(
        {
            "description": [
                "The quarterly budget review meeting is scheduled for Friday morning.",
                "Please confirm that all traders are set up correctly before go-live.",
            ],
            "actual_subject": ["Budget Review\n", "Trader Setup\n"],
        }
    )
    module = _load_run_benchmark()
    monkeypatch.setattr(module, "load_aeslc_sample", lambda **kwargs: frame.copy())
    return module


def test_skip_model_benchmark_writes_metrics(fake_aeslc, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--samples",
            "2",
            "--skip-model",
            "--output-dir",
            str(tmp_path),
        ],
    )
    fake_aeslc.main()

    metrics_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "generated_subjects_aeslc_benchmark.csv"
    assert metrics_path.is_file()
    assert csv_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["n"] == 2
    assert metrics["primary_method"] == "naive"
    for key in ("naive", "zero_shot", "best_of_n"):
        assert "rouge1" in metrics["methods"][key]
        assert "rougeL" in metrics["methods"][key]
    assert Path(csv_path).stat().st_size > 0
