"""Paths, model name, and decoding defaults."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

MODEL_NAME = "distilgpt2"
SEED = 42
SAFE_MAX_WORDS = 5
AESLC_SAMPLE_SIZE = 100
AESLC_SEED = 42
AESLC_BODY_WORDS = 50
CANDIDATES_PER_DESC = 3
NUM_VARIANTS = 1

MAX_NEW_TOKENS = 16
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9
MAX_TRIES = 3


def project_root() -> Path:
    """Resolve the repo root from this file or the current working directory."""
    start = Path(__file__).resolve().parent
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "src").is_dir():
            return path
    cwd = Path.cwd()
    if (cwd / "src").is_dir() and (cwd / "src" / "email_subject_generator").is_dir():
        return cwd
    return cwd


PROJECT_ROOT = project_root()
OUTPUT_DIR = PROJECT_ROOT / "outputs"


BASE_TEMPLATE = (
    "You are a professional email copywriter.\n"
    "Write ONE short email subject line.\n"
    "Rules: <= 5 words, Title Case, no emojis/quotes,\n"
    "no links, no brackets/parentheses, no markdown, no code,\n"
    "no first-person pronouns (I, I'm, my, we, us, our),\n"
    "avoid 'Re:' and 'Fwd:'.\n\n"
    "Description: {desc}\n"
    "Subject:"
)
