"""Generator tests use a fake pipeline so CI never downloads DistilGPT-2."""

from email_subject_generator.generator import SubjectGenerator


class _FakeTokenizer:
    eos_token_id = 50256

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 50 + 1 for ch in str(text)[:4]] or [1]


class _FakePipeline:
    tokenizer = _FakeTokenizer()

    def __init__(self, suffix: str = "Budget Review Meeting") -> None:
        self.suffix = suffix
        self.calls = 0

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        return [{"generated_text": f"{prompt}{self.suffix}"}]


def test_generate_subject_returns_nonempty():
    gen = SubjectGenerator(pipeline=_FakePipeline())
    subject = gen.generate_subject("Please review the attached quarterly budget.")
    assert isinstance(subject, str)
    assert len(subject.strip()) >= 2


def test_best_of_n_returns_requested_count():
    gen = SubjectGenerator(pipeline=_FakePipeline())
    subjects = gen.best_of_n(
        "Team offsite planning for next Thursday afternoon",
        n=4,
        k=2,
        min_overlap=0,
    )
    assert 1 <= len(subjects) <= 2
    assert all(isinstance(s, str) and s.strip() for s in subjects)


def test_best_of_n_fallback_when_empty_candidates():
    class EmptyPipeline:
        tokenizer = _FakeTokenizer()

        def __call__(self, prompt, **kwargs):
            return [{"generated_text": prompt}]

    gen = SubjectGenerator(pipeline=EmptyPipeline())
    subjects = gen.best_of_n("Quarterly budget review meeting tomorrow", n=2, k=1, min_overlap=0)
    assert subjects
    assert subjects[0]
