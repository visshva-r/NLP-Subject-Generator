"""AESLC dataset loading helpers."""

from __future__ import annotations

import pandas as pd

from .config import AESLC_BODY_WORDS, AESLC_SAMPLE_SIZE, AESLC_SEED


def truncate_body(text: str, max_words: int = AESLC_BODY_WORDS) -> str:
    return " ".join(str(text).split()[:max_words])


def load_aeslc_sample(
    n: int = AESLC_SAMPLE_SIZE,
    seed: int = AESLC_SEED,
    max_words: int = AESLC_BODY_WORDS,
    split: str = "test",
) -> pd.DataFrame:
    """Load a reproducible AESLC sample with truncated email bodies."""
    from datasets import load_dataset

    dataset = load_dataset("aeslc", split=split)
    raw = dataset.to_pandas()
    if n < len(raw):
        raw = raw.sample(n, random_state=seed).reset_index(drop=True)
    else:
        raw = raw.reset_index(drop=True)

    return pd.DataFrame(
        {
            "description": raw["email_body"].astype(str).map(lambda x: truncate_body(x, max_words)),
            "actual_subject": raw["subject_line"].astype(str),
        }
    )
