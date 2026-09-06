"""DistilGPT-2 subject generation with retry filters and best-of-N overlap scoring."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .config import (
    BASE_TEMPLATE,
    MAX_NEW_TOKENS,
    MAX_TRIES,
    MODEL_NAME,
    SAFE_MAX_WORDS,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from .preprocessing import (
    BAD_ENDINGS,
    BAD_STRINGS,
    INSTRUCTION_TOKENS,
    NEG_TOKENS,
    clean_subject,
    keyword_set,
)

PipelineFn = Callable[..., Any]


def _device_id() -> int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _encode_bad_words(tokenizer: Any) -> list[list[int]]:
    ids: list[list[int]] = []
    for token in BAD_STRINGS:
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                ids.append(encoded)
        except Exception:
            continue
    return ids


class SubjectGenerator:
    """Wraps a Hugging Face text-generation pipeline (or a test double)."""

    def __init__(
        self,
        pipeline: Optional[PipelineFn] = None,
        model_name: str = MODEL_NAME,
        device: Optional[int] = None,
        seed: int = SEED,
    ) -> None:
        self.model_name = model_name
        self.device = _device_id() if device is None else device
        if pipeline is None:
            from transformers import pipeline as hf_pipeline, set_seed
            import torch

            set_seed(seed)
            if torch.cuda.is_available():
                torch.manual_seed(seed)
            pipeline = hf_pipeline(
                "text-generation",
                model=model_name,
                framework="pt",
                device=self.device,
            )
        self.pipeline = pipeline
        tokenizer = getattr(pipeline, "tokenizer", None)
        self.bad_words_ids = _encode_bad_words(tokenizer) if tokenizer is not None else None
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    def generate_subject(
        self,
        description: str,
        template: str = BASE_TEMPLATE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
        top_p: float = TOP_P,
        max_tries: int = MAX_TRIES,
        max_words: int = SAFE_MAX_WORDS,
    ) -> str:
        prompt = template.format(desc=(description or "").strip())
        last = ""
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "num_return_sequences": 1,
            "pad_token_id": 50256,
        }
        if self.bad_words_ids:
            gen_kwargs["bad_words_ids"] = self.bad_words_ids
        if self.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.eos_token_id

        for _ in range(max_tries):
            out = self.pipeline(prompt, **gen_kwargs)[0]["generated_text"]
            gen = out.replace(prompt, "")
            cleaned = clean_subject(gen, fallback=description, max_words=max_words)
            last = cleaned

            words = cleaned.lower().split()
            if any(t in words for t in INSTRUCTION_TOKENS):
                continue
            if any(n in words for n in NEG_TOKENS) and not (keyword_set(description) & NEG_TOKENS):
                continue
            if len(cleaned.split()) >= 2 and cleaned.split()[-1].lower() not in BAD_ENDINGS:
                return cleaned

        return last if last else clean_subject("", fallback=description, max_words=max_words)

    def best_of_n(
        self,
        description: str,
        n: int = 10,
        k: int = 2,
        min_overlap: int = 1,
        **generate_kwargs: Any,
    ) -> list[str]:
        cands: list[str] = []
        seen: set[str] = set()
        for _ in range(n):
            subject = self.generate_subject(description, **generate_kwargs)
            key = subject.lower()
            if key not in seen:
                seen.add(key)
                cands.append(subject)

        desc_keywords = keyword_set(description)

        def score(subj: str) -> tuple[int, float]:
            overlap = len(desc_keywords & keyword_set(subj))
            return (overlap, 0.05 * len(keyword_set(subj)))

        cands.sort(key=score, reverse=True)

        selected: list[str] = []
        for subject in cands:
            if score(subject)[0] >= min_overlap:
                selected.append(subject)
            if len(selected) == k:
                break

        if len(selected) < k:
            for subject in cands:
                if subject not in selected:
                    selected.append(subject)
                if len(selected) == k:
                    break

        if not selected:
            selected = [
                clean_subject(
                    "",
                    fallback=description,
                    max_words=generate_kwargs.get("max_words", SAFE_MAX_WORDS),
                )
            ]
        return selected


_default: Optional[SubjectGenerator] = None


def load_generator(
    pipeline: Optional[PipelineFn] = None,
    model_name: str = MODEL_NAME,
    device: Optional[int] = None,
    seed: int = SEED,
) -> SubjectGenerator:
    global _default
    _default = SubjectGenerator(
        pipeline=pipeline, model_name=model_name, device=device, seed=seed
    )
    return _default


def _require_default() -> SubjectGenerator:
    if _default is None:
        return load_generator()
    return _default


def generate_subject(description: str, **kwargs: Any) -> str:
    return _require_default().generate_subject(description, **kwargs)


def best_of_n(description: str, n: int = 10, k: int = 2, min_overlap: int = 1, **kwargs: Any) -> list[str]:
    return _require_default().best_of_n(description, n=n, k=k, min_overlap=min_overlap, **kwargs)
