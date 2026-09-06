"""Generate one or more subject lines for a single email body."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_subject_generator.config import CANDIDATES_PER_DESC, MODEL_NAME, NUM_VARIANTS  # noqa: E402
from email_subject_generator.generator import SubjectGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate email subject lines")
    parser.add_argument("text", nargs="?", help="Email body. If omitted, read stdin.")
    parser.add_argument("--candidates", type=int, default=max(CANDIDATES_PER_DESC, 3))
    parser.add_argument("--k", type=int, default=max(NUM_VARIANTS, 3), help="How many subjects to print")
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text else sys.stdin.read()
    text = (text or "").strip()
    if not text:
        raise SystemExit("Provide an email body as an argument or via stdin.")

    print(f"Loading {MODEL_NAME}...")
    gen = SubjectGenerator()
    subjects = gen.best_of_n(
        text,
        n=args.candidates,
        k=args.k,
        min_overlap=0,
        temperature=args.temperature,
    )
    print("\nSuggested subject lines:")
    for i, subject in enumerate(subjects, start=1):
        print(f"  {i}. {subject}")


if __name__ == "__main__":
    main()
