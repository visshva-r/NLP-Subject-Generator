"""Tests for subject cleanup, naive baseline, and keyword helpers."""

from email_subject_generator.preprocessing import clean_subject, keyword_set, naive_subject


def test_clean_subject_empty_uses_fallback():
    result = clean_subject("", fallback="Quarterly budget review meeting tomorrow")
    assert result
    assert len(result.split()) >= 2


def test_clean_subject_strips_punctuation_and_links():
    raw = "Subject: Re: **Q3 Update** — see https://example.com (draft)"
    cleaned = clean_subject(raw, fallback="Q3 product update for customers")
    assert "http" not in cleaned.lower()
    assert "*" not in cleaned
    assert cleaned


def test_clean_subject_caps_length():
    long_text = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel"
    cleaned = clean_subject(long_text)
    assert len(cleaned.split()) <= 5


def test_clean_subject_respects_max_words():
    long_text = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel"
    cleaned = clean_subject(long_text, max_words=3)
    assert len(cleaned.split()) <= 3


def test_clean_subject_drops_first_person():
    cleaned = clean_subject("I we our Meeting Agenda", fallback="Team meeting agenda notes")
    lowered = set(cleaned.lower().split())
    assert "i" not in lowered
    assert "we" not in lowered
    assert "our" not in lowered


def test_naive_subject_takes_content_words():
    subject = naive_subject("The quarterly budget review is scheduled for Friday")
    assert subject
    assert "The" not in subject.split()[:1] or subject.startswith("Quarterly")
    assert "quarterly" in subject.lower() or "budget" in subject.lower()


def test_keyword_set_drops_stopwords():
    keys = keyword_set("The meeting about the budget")
    assert "the" not in keys
    assert "meeting" in keys
    assert "budget" in keys
