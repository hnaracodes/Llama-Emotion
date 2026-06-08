"""Benchmark helpers for affective modulation evaluation."""

from src.benchmark.affect_metrics import (
    cosine_distance,
    empathy_lexical_score,
    kl_divergence_from_logits,
    last_token_logits,
    sentiment_lexical_score,
    text_hidden_embedding,
)

__all__ = [
    "cosine_distance",
    "empathy_lexical_score",
    "kl_divergence_from_logits",
    "last_token_logits",
    "sentiment_lexical_score",
    "text_hidden_embedding",
]
