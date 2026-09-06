"""Tests for AESLC helpers that do not require a network download."""

from email_subject_generator.data import truncate_body


def test_truncate_body_caps_word_count():
    text = " ".join(f"word{i}" for i in range(80))
    truncated = truncate_body(text, max_words=50)
    assert len(truncated.split()) == 50
