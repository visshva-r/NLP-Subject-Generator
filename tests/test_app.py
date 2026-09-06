"""Gradio app tests — no DistilGPT-2 download."""

from app import build_demo, launch_demo, suggest_subjects
from email_subject_generator.generator import SubjectGenerator


class _FakeTokenizer:
    eos_token_id = 50256

    def encode(self, text, add_special_tokens=False):
        return [1]


class _FakePipeline:
    tokenizer = _FakeTokenizer()

    def __call__(self, prompt, **kwargs):
        return [{"generated_text": f"{prompt}Budget Review Meeting"}]


def test_suggest_subjects_requires_body():
    baseline, model = suggest_subjects("")
    assert baseline == "-"
    assert "Paste" in model


def test_suggest_subjects_with_fake_generator():
    gen = SubjectGenerator(pipeline=_FakePipeline())
    baseline, model = suggest_subjects(
        "Please review the attached quarterly budget for Friday.",
        temperature=0.7,
        num_candidates=3,
        max_words=5,
        num_suggestions=2,
        generator=gen,
    )
    assert "budget" in baseline.lower() or "quarterly" in baseline.lower()
    assert "**1.**" in model
    assert "Budget" in model or "Review" in model


def test_build_demo_constructs_interface():
    demo = build_demo()
    assert demo is not None


def test_launch_demo_is_callable():
    assert callable(launch_demo)
