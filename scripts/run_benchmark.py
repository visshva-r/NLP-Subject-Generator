"""Run AESLC subject-generation benchmark and write CSV + metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from email_subject_generator.config import (  # noqa: E402
    AESLC_SAMPLE_SIZE,
    AESLC_SEED,
    CANDIDATES_PER_DESC,
    MODEL_NAME,
    NUM_VARIANTS,
    OUTPUT_DIR,
)
from email_subject_generator.data import load_aeslc_sample  # noqa: E402
from email_subject_generator.evaluation import pack_method_metrics  # noqa: E402
from email_subject_generator.generator import SubjectGenerator  # noqa: E402
from email_subject_generator.preprocessing import naive_subject  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AESLC email-subject benchmark")
    parser.add_argument("--samples", type=int, default=AESLC_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=AESLC_SEED)
    parser.add_argument("--candidates", type=int, default=CANDIDATES_PER_DESC)
    parser.add_argument("--variants", type=int, default=NUM_VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip DistilGPT-2 and only evaluate the naive extractive baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AESLC test sample (n={args.samples}, seed={args.seed})...")
    df = load_aeslc_sample(n=args.samples, seed=args.seed)
    refs = df["actual_subject"].tolist()

    df["naive_subject"] = df["description"].map(naive_subject)
    methods = {"naive": pack_method_metrics(df["naive_subject"].tolist(), refs)}

    if args.skip_model:
        df["zero_shot_subject"] = df["naive_subject"]
        df["generated_subject"] = df["naive_subject"]
        methods["zero_shot"] = methods["naive"]
        methods["best_of_n"] = methods["naive"]
        primary = "naive"
    else:
        print(f"Loading {MODEL_NAME}...")
        gen = SubjectGenerator(seed=args.seed)
        from tqdm import tqdm

        zero_shot = []
        for desc in tqdm(df["description"], total=len(df), desc="Zero-shot"):
            zero_shot.append(gen.generate_subject(desc))
        df["zero_shot_subject"] = zero_shot
        methods["zero_shot"] = pack_method_metrics(zero_shot, refs)

        best = []
        for desc in tqdm(df["description"], total=len(df), desc="Best-of-N"):
            best.append(
                gen.best_of_n(desc, n=args.candidates, k=args.variants, min_overlap=0)[0]
            )
        df["generated_subject"] = best
        methods["best_of_n"] = pack_method_metrics(best, refs)
        primary = "best_of_n"

    df["len_words"] = df["generated_subject"].str.split().map(len)

    metrics = {
        "n": int(args.samples),
        "seed": int(args.seed),
        "model": MODEL_NAME,
        "candidates": int(args.candidates),
        "primary_method": primary,
        "methods": methods,
        "protocol": (
            f"AESLC test split, sample n={args.samples}, seed={args.seed}, "
            "email body truncated to 50 words. Naive = first 5 content words, title-cased. "
            f"Zero-shot = single DistilGPT-2 sample. Best-of-N = {args.candidates} samples "
            "ranked by keyword overlap with the email body."
        ),
    }

    csv_path = output_dir / "generated_subjects_aeslc_benchmark.csv"
    json_path = output_dir / "metrics.json"
    rows_path = output_dir / "generated_subjects_aeslc_benchmark.json"

    export_cols = [
        "description",
        "actual_subject",
        "naive_subject",
        "zero_shot_subject",
        "generated_subject",
        "len_words",
    ]
    df[export_cols].to_csv(csv_path, index=False)
    df[export_cols].assign(description=lambda d: d["description"].str.slice(0, 100) + "...").to_json(
        rows_path, orient="records", indent=2
    )
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n--- AESLC comparison (ROUGE vs human subject lines) ---")
    for name, block in methods.items():
        print(
            f"{name:10}  R1={block['rouge1']:.4f}  "
            f"R2={block['rouge2']:.4f}  RL={block['rougeL']:.4f}  "
            f"len={block['avg_subject_length']:.2f}  uniq={block['unique_ratio']:.2%}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
