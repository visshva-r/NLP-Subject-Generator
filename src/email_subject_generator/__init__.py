"""Email subject line generation with DistilGPT-2 and best-of-N scoring."""

from .config import MODEL_NAME, OUTPUT_DIR, PROJECT_ROOT, SAFE_MAX_WORDS
from .preprocessing import clean_subject, keyword_set, naive_subject
from .generator import SubjectGenerator, best_of_n, generate_subject, load_generator
from .evaluation import compute_rouge, pack_method_metrics, summarize_metrics
from .data import load_aeslc_sample

__all__ = [
    "MODEL_NAME",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "SAFE_MAX_WORDS",
    "SubjectGenerator",
    "best_of_n",
    "clean_subject",
    "compute_rouge",
    "generate_subject",
    "keyword_set",
    "load_aeslc_sample",
    "load_generator",
    "naive_subject",
    "pack_method_metrics",
    "summarize_metrics",
]
