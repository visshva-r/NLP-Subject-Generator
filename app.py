"""Gradio demo for extractive vs DistilGPT-2 best-of-N email subject generation."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from email_subject_generator.config import MODEL_NAME, SAFE_MAX_WORDS, TEMPERATURE  # noqa: E402
from email_subject_generator.generator import SubjectGenerator  # noqa: E402
from email_subject_generator.preprocessing import naive_subject  # noqa: E402

SAMPLE_EMAILS = [
    "Last week was the hardest week that our office has ever experienced. "
    "When I look down the list of people who were laid off, I see friends and colleagues.",
    "Could you please coordinate with Sara and Richard's assistant to set up a 30 minute meeting "
    "next week to discuss the Q3 budget review?",
    "Attached is a draft letter agreement under which Transwestern would provide interruptible "
    "transportation service beginning next month.",
]

_CSS = """
.panel { min-height: 7rem; }
footer { display: none !important; }
"""


@lru_cache(maxsize=1)
def get_generator() -> SubjectGenerator:
    return SubjectGenerator()


def _format_ranked(subjects: list[str]) -> str:
    lines = [s.strip() for s in subjects if s and s.strip()]
    if not lines:
        return "_No subject generated. Try a longer body._"
    return "\n\n".join(f"**{i}.** {line}" for i, line in enumerate(lines, start=1))


def suggest_subjects(
    email_body: str,
    temperature: float = TEMPERATURE,
    num_candidates: int = 3,
    max_words: int = SAFE_MAX_WORDS,
    num_suggestions: int = 3,
    generator: Optional[SubjectGenerator] = None,
) -> tuple[str, str]:
    """Return (extractive baseline, ranked model subjects)."""
    text = (email_body or "").strip()
    if not text:
        return ("-", "Paste an email first.")

    words = max(2, min(int(max_words), 12))
    n = max(1, int(num_candidates))
    k = max(1, min(int(num_suggestions), n, 3))

    baseline = naive_subject(text, max_words=words)
    gen = generator or get_generator()
    ranked = gen.best_of_n(
        text,
        n=n,
        k=k,
        min_overlap=0,
        temperature=float(temperature),
        max_words=words,
    )
    return baseline or "-", _format_ranked(ranked)


def build_demo() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="zinc",
        secondary_hue="zinc",
        neutral_hue="stone",
    ).set(
        button_primary_background_fill="#1c1917",
        button_primary_background_fill_hover="#292524",
        button_primary_text_color="#fafaf9",
        block_border_width="1px",
    )

    with gr.Blocks(title="Email Subject Line Generator", theme=theme, css=_CSS) as demo:
        gr.Markdown(
            """
# Email subject line generator

Paste an email body to compare two subject drafts:
an extractive baseline (first content words) and DistilGPT-2 with best-of-N ranking.
"""
        )
        email = gr.Textbox(
            label="Email body",
            lines=8,
            placeholder="Paste the email body here",
        )
        run = gr.Button("Generate subjects", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Extractive baseline")
                baseline_out = gr.Markdown(value="_Waiting for input._", elem_classes=["panel"])
            with gr.Column():
                gr.Markdown("DistilGPT-2 (best of 3)")
                model_out = gr.Markdown(value="_Waiting for input._", elem_classes=["panel"])
        with gr.Accordion("Decoding options", open=False):
            temperature = gr.Slider(0.1, 1.2, value=TEMPERATURE, step=0.05, label="Temperature")
            num_candidates = gr.Slider(1, 8, value=3, step=1, label="Candidates sampled")
            max_words = gr.Slider(3, 8, value=SAFE_MAX_WORDS, step=1, label="Max words")
            num_suggestions = gr.Slider(1, 3, value=3, step=1, label="Suggestions to keep")
        gr.Examples(examples=[[s] for s in SAMPLE_EMAILS], inputs=[email], label="Examples")
        gr.Markdown(
            f"Model: `{MODEL_NAME}`. First run downloads weights (~80 MB). "
            "Benchmark metrics are documented in the repository README."
        )
        run.click(
            fn=suggest_subjects,
            inputs=[email, temperature, num_candidates, max_words, num_suggestions],
            outputs=[baseline_out, model_out],
        )
        email.submit(
            fn=suggest_subjects,
            inputs=[email, temperature, num_candidates, max_words, num_suggestions],
            outputs=[baseline_out, model_out],
        )
    return demo


def launch_demo(demo: gr.Blocks) -> None:
    share = os.getenv("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}
    in_space = bool(os.getenv("SPACE_ID"))
    server_name = os.getenv("GRADIO_SERVER_NAME")
    if in_space:
        server_name = server_name or "0.0.0.0"
    elif not server_name:
        server_name = "127.0.0.1"

    kwargs = {"share": share, "inbrowser": False, "server_name": server_name}
    try:
        demo.launch(**kwargs)
    except ValueError as exc:
        text = str(exc).lower()
        if "localhost is not accessible" in text or "shareable link" in text:
            print("Localhost was not reachable; retrying with a Gradio share link.")
            demo.launch(share=True, inbrowser=False, server_name=server_name)
        else:
            raise


if __name__ == "__main__":
    launch_demo(build_demo())
