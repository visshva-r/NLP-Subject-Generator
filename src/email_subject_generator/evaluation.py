"""ROUGE evaluation and summary stats for generated subject lines."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    import evaluate

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=list(predictions), references=list(references))
    return {
        "rouge1": float(results["rouge1"]),
        "rouge2": float(results["rouge2"]),
        "rougeL": float(results["rougeL"]),
    }


def summarize_metrics(
    generated: Iterable[str],
    rouge: Mapping[str, float] | None = None,
) -> dict[str, float]:
    series = pd.Series(list(generated), dtype="object").fillna("")
    lengths = series.str.split().map(len)
    n = max(len(series), 1)
    summary = {
        "n": float(len(series)),
        "avg_subject_length": float(lengths.mean()) if len(series) else 0.0,
        "unique_ratio": float(series.nunique() / n),
    }
    if rouge:
        summary.update({k: float(v) for k, v in rouge.items()})
    return summary


def pack_method_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """ROUGE + length/uniqueness for one generation method."""
    rouge = compute_rouge(predictions, references)
    return summarize_metrics(predictions, rouge)

