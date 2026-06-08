"""Metrics for Phase 4 extended and chat A/B affect benchmarks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_EMPATHY_TERMS = (
    "sorry",
    "understand",
    "here for you",
    "with you",
    "support",
    "difficult",
    "hard",
    "care",
    "listen",
    "valid",
    "feel",
    "hope",
    "help",
    "gentle",
    "kind",
)

_POSITIVE_TERMS = (
    "good",
    "great",
    "hope",
    "better",
    "glad",
    "proud",
    "warm",
    "kind",
    "calm",
    "peace",
    "thank",
    "appreciate",
    "encourage",
)

_NEGATIVE_TERMS = (
    "bad",
    "awful",
    "terrible",
    "hopeless",
    "worthless",
    "fail",
    "hate",
    "angry",
    "upset",
    "wrong",
    "never",
    "can't",
    "cannot",
)


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def empathy_lexical_score(text: str) -> float:
    """
    Normalized empathy-marker density in [0, 1].

    Heuristic only — useful for relative A/B comparison, not ground-truth empathy.
    """
    words = max(1, len(text.split()))
    hits = _count_terms(text, _EMPATHY_TERMS)
    return round(min(1.0, hits / (words * 0.15 + 1.0)), 4)


def sentiment_lexical_score(text: str) -> float:
    """
    Simple lexical sentiment in [-1, 1] from positive vs negative term counts.
    """
    pos = _count_terms(text, _POSITIVE_TERMS)
    neg = _count_terms(text, _NEGATIVE_TERMS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


@torch.inference_mode()
def last_token_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    """Logits at the last prompt token position, shape (vocab,)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.logits[0, -1, :].float()


def kl_divergence_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) over vocabulary."""
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    kl = (p * (p.clamp(min=1e-12).log() - q.clamp(min=1e-12).log())).sum()
    return round(float(kl.item()), 6)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine similarity between two vectors."""
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return round(1.0 - sim, 6)


@torch.inference_mode()
def text_hidden_embedding(model, tokenizer, text: str) -> torch.Tensor:
    """Mean-pooled last-layer hidden state as a lightweight text embedding."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    return hidden.mean(dim=1).squeeze(0)


def compare_generations(
    text_a: str,
    text_b: str,
    *,
    model=None,
    tokenizer=None,
) -> dict[str, Any]:
    """Lexical + optional model-embedding deltas between two generations."""
    metrics: dict[str, Any] = {
        "text_changed": text_a.strip() != text_b.strip(),
        "empathy_a": empathy_lexical_score(text_a),
        "empathy_b": empathy_lexical_score(text_b),
        "empathy_delta": round(
            empathy_lexical_score(text_b) - empathy_lexical_score(text_a), 4
        ),
        "sentiment_a": sentiment_lexical_score(text_a),
        "sentiment_b": sentiment_lexical_score(text_b),
        "sentiment_delta": round(
            sentiment_lexical_score(text_b) - sentiment_lexical_score(text_a), 4
        ),
        "length_delta_chars": len(text_b) - len(text_a),
    }
    if model is not None and tokenizer is not None:
        emb_a = text_hidden_embedding(model, tokenizer, text_a)
        emb_b = text_hidden_embedding(model, tokenizer, text_b)
        metrics["embedding_cosine_distance"] = cosine_distance(emb_a, emb_b)
    return metrics
