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


def test_compare_generations_full_text_across_scripted_transcripts_is_misleading():
    """Regression for a bug found in `benchmark_phase_chat_ab.py`: its
    cross-scenario comparisons (`comparisons_vs_neutral`, `distress_vs_hopeful`)
    fed `compare_generations` the *full* decoded sequence (scripted transcript
    + generation) for two DIFFERENT scenarios, each with its own scripted
    dialogue. Unlike hooks-on/off comparisons elsewhere in the codebase (which
    reuse one identical prompt, so the shared prefix cancels out), chat A/B's
    scenarios have distinct scripted prompts — comparing full text there
    measures differences in the benchmark author's own scripted words, not
    the model's actual generated behavior. This test documents why the fix
    (comparing `stats["new_text"]` only) is required: identical model output
    wrapped in different scripted prompts must compare as unchanged.
    """
    neutral_prompt = "Hi, I'm planning my week. Sure, what are your priorities?"
    distress_prompt = "I failed my exam and I feel awful. I'm here with you and I care."
    same_reply = "Take it one step at a time and be kind to yourself."

    full_text_cmp = compare_generations(
        neutral_prompt + same_reply, distress_prompt + same_reply
    )
    assert full_text_cmp["text_changed"] is True
    assert full_text_cmp["empathy_delta"] != 0.0

    generated_only_cmp = compare_generations(same_reply, same_reply)
    assert generated_only_cmp["text_changed"] is False
    assert generated_only_cmp["empathy_delta"] == 0.0


def test_coupling_corr_from_affect_metrics():
    from src.affective.coupling import affect_coupling_corr
    from src.affective.emotion_lexicon import emotion_to_32d

    t = [emotion_to_32d("sad"), emotion_to_32d("anxious"), emotion_to_32d("afraid")]
    assert affect_coupling_corr(t, t) >= 0.99
    rand = [emotion_to_32d("grateful"), emotion_to_32d("afraid")]
    assert affect_coupling_corr(t, rand) < affect_coupling_corr(t, t)
