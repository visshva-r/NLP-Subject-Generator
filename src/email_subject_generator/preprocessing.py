"""Subject-line cleanup, keyword overlap helpers, and a naive extractive baseline."""

from __future__ import annotations

import re
import unicodedata

from .config import SAFE_MAX_WORDS

BAD_ENDINGS = {
    "a", "an", "the", "of", "to", "for", "and", "in", "with", "on", "at", "by",
    "or", "is", "this", "be", "are",
}
FORBID_PRONOUNS = {"i", "i'm", "im", "me", "my", "mine", "we", "us", "our", "ours"}
NEG_TOKENS = {"not", "no", "never", "sorry", "can't", "won't", "don't", "cannot"}
ACRONYMS = {"ai", "ml", "nlp"}
INSTRUCTION_TOKENS = {"title", "write", "subject", "line", "description", "topic", "brief", "one"}

STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "and", "in", "with", "on", "at", "by", "from",
    "is", "are", "be", "this", "that", "your", "our", "my", "we", "you", "about", "email",
}

BAD_STRINGS = {
    "http", "www", "href", "mailto", "re", "re:", "fwd", "fwd:",
    "<", ">", "[", "]", "(", ")", "`", "*", "#", "|", "\\", "_", "=",
    "hello", "please", "thanks", "thank", "emoji", "yes",
    "write", "subject", "line", "description", "topic", "brief", "one",
    "hey", "title", "guys", "ve",
}
BAD_STRINGS_LOWER = {b.lower() for b in BAD_STRINGS}


def titleize(tokens: list[str]) -> str:
    out = []
    for i, token in enumerate(tokens):
        low = token.lower()
        if low in ACRONYMS:
            out.append(low.upper())
        elif i == 0 or len(token) >= 3:
            out.append(token.capitalize())
        else:
            out.append(low)
    return " ".join(out)


def keyword_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b\w+\b", text) if w.lower() not in STOPWORDS}


def naive_subject(description: str, max_words: int = SAFE_MAX_WORDS) -> str:
    """Title-cased first N content words — a simple extractive baseline."""
    tokens = re.sub(r"[^A-Za-z0-9\s]", " ", description or "")
    tokens = re.sub(r"\s+", " ", tokens).strip().split()
    content = [w for w in tokens if w.lower() not in STOPWORDS][:max_words]
    if not content:
        return "Email Update"
    return titleize(content)


def clean_subject(s: str, fallback: str = "", max_words: int = SAFE_MAX_WORDS) -> str:
    s = (s or "").split("\n")[0].replace("Subject:", "")
    s = re.sub(r"(?:re:|fwd:)\s*", "", s, flags=re.I)
    s = re.sub(r"http\S+|www\.\S+|mailto:\S+", "", s)
    s = re.sub(r'[\[\]()<>`"*#~|\\=/?:]', " ", s)
    s = s.strip(" \"'")
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks = []
    for token in s.split():
        low = token.lower()
        if token.isdigit() or any(ch.isdigit() for ch in token):
            continue
        if not token.isalpha():
            continue
        if low in FORBID_PRONOUNS or low in BAD_STRINGS_LOWER:
            continue
        if len(token) == 1:
            continue
        toks.append(token)

    dedup = []
    for token in toks:
        if not dedup or token.lower() != dedup[-1].lower():
            dedup.append(token)
    cap = max(1, int(max_words))
    toks = dedup[:cap]

    if len(toks) < 2 or (toks and toks[-1].lower() in BAD_ENDINGS):
        fb = re.sub(r"[^A-Za-z0-9\s]", " ", fallback)
        fb = re.sub(r"\s+", " ", fb).strip().split()
        fb = [w for w in fb if w.lower() not in STOPWORDS][:cap]
        if len(fb) >= 2:
            toks = fb

    return titleize(toks).rstrip(" .,!?:;-/\\")
