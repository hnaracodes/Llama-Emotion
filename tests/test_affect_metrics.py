"""Unit tests for affect benchmark metrics (CPU, no GPU)."""

import torch

from src.benchmark.affect_metrics import (
    compare_generations,
    cosine_distance,
    empathy_lexical_score,
    kl_divergence_from_logits,
    sentiment_lexical_score,
)


def test_empathy_lexical_score_increases_with_markers():
    low = "Photosynthesis converts light to chemical energy."
    high = "I'm sorry you're going through this. I understand and I'm here for you."
    assert empathy_lexical_score(high) > empathy_lexical_score(low)


def test_sentiment_lexical_score_sign():
    pos = "I'm glad things are better and hopeful today."
    neg = "This is awful and hopeless."
    assert sentiment_lexical_score(pos) > 0
    assert sentiment_lexical_score(neg) < 0


def test_kl_divergence_zero_for_identical_logits():
    logits = torch.randn(100)
    assert kl_divergence_from_logits(logits, logits) == 0.0


def test_kl_divergence_positive_for_different_logits():
    a = torch.zeros(10)
    a[0] = 10.0
    b = torch.zeros(10)
    b[1] = 10.0
    assert kl_divergence_from_logits(a, b) > 0.0


def test_cosine_distance_identical_vectors():
    v = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == 0.0


def test_compare_generations_lexical_only():
    cmp = compare_generations("hello world", "hello there")
    assert "text_changed" in cmp
    assert "empathy_delta" in cmp
    assert "embedding_cosine_distance" not in cmp
