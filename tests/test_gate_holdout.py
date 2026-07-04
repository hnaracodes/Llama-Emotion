"""Gate holdout collapse detection tests."""

from src.benchmark.gate_holdout import (
    collapse_score,
    detect_empathy_collapse,
    detect_glued_empathy_morphs,
    holdout_prompts,
)
from src.config import GATE_TRAIN_HOLDOUT_PROMPTS, PHASE4_ABLATION_PROMPTS


def test_detect_feelfeel_collapse():
    text = "I feel feel feel feel feel feel feel feel feel sorry"
    assert detect_empathy_collapse(text) is True


def test_coherent_reply_not_collapsed():
    text = "That sounds really difficult. What happened at work today?"
    assert detect_empathy_collapse(text) is False
    assert collapse_score(text) < 0.5


def test_sorry_run_length_collapse():
    text = "sorry " * 12
    assert detect_empathy_collapse(text) is True


def test_hereunderstand_morph_collapse():
    text = "hereunderstandhereunderstandhereunderstandhereunderstand"
    assert detect_glued_empathy_morphs(text) is True
    assert detect_empathy_collapse(text) is True


def test_collapse_score_ranking():
    bad = collapse_score("carecarecarecarecare")
    good = collapse_score("Photosynthesis converts light to chemical energy.")
    assert bad > good


# --- Regression tests against real v1/v2 collapse transcripts -------------
# These exact strings were observed in production Gate v1/v2 holdout logs
# before v3's listener-CE rewrite. Any future change to the detector must
# keep flagging all of them.


def test_regression_v1_sorry_loop():
    text = " ".join(["sorry"] * 20)
    assert detect_empathy_collapse(text) is True


def test_regression_v2_glued_morph_chain():
    text = "hereunderstandhereunderstandhereunderstandhereunderstandhereunderstand"
    # detect_empathy_collapse flags glued morphs directly even when the
    # composite collapse_score alone (0.45 here) falls short of its own 0.5
    # threshold — the two checks are intentionally not equivalent.
    assert detect_empathy_collapse(text) is True
    assert collapse_score(text) >= 0.4


def test_regression_mixed_empathy_word_salad():
    text = "care feel sorry help understand care feel sorry help understand care"
    assert detect_empathy_collapse(text) is True


def test_regression_low_diversity_non_glued_repeat():
    """Low unique-word-ratio text that isn't a clean glued morph or regex
    hit should still be flagged via the collapse_score composite."""
    text = ("thing thing thing stuff thing stuff thing stuff thing stuff " * 2).strip()
    assert collapse_score(text) >= 0.25


# --- v3.1 hardening: training-time holdout prompts must not leak into ----
# --- the Phase 4 ablation set used for independent post-hoc validation ----


def test_gate_holdout_prompts_disjoint_from_phase4_ablation():
    """Checkpoint selection must not use the same prompts as the report
    that later claims to independently validate the saved checkpoint.
    """
    holdout_ids = {p["id"] for p in holdout_prompts()}
    phase4_ids = {p["id"] for p in PHASE4_ABLATION_PROMPTS}
    assert holdout_ids.isdisjoint(phase4_ids)

    holdout_prompt_texts = {p["prompt"] for p in holdout_prompts()}
    phase4_prompt_texts = {p["prompt"] for p in PHASE4_ABLATION_PROMPTS}
    assert holdout_prompt_texts.isdisjoint(phase4_prompt_texts)


def test_gate_holdout_prompts_matches_config():
    assert holdout_prompts() == list(GATE_TRAIN_HOLDOUT_PROMPTS)


# --- v3.1 hardening: long coherent text must not false-positive ------------
# Regression for a bug found in the v3.1 behavioral verification run: the
# low-diversity fallback branch of `detect_glued_empathy_morphs` compared
# `len(set(compact))` (bounded by the 26-letter alphabet) against
# `len(compact) // 4`, which is trivially true for *any* text longer than
# ~104 characters. That made `detect_empathy_collapse` fire on genuinely
# coherent, multi-sentence empathetic replies that simply used "feel" /
# "care" / "here" / "understand" a handful of times across separate
# sentences — nothing like true glued-morpheme repetition collapse.


def test_long_coherent_empathetic_reply_not_collapsed():
    text = (
        "It's normal to feel that way after a tough test. Allow yourself to "
        "grieve: give yourself permission to process your emotions. It's "
        "okay to feel sad, frustrated, or disappointed. Take care of "
        "yourself: make sure you are eating, sleeping, and doing things you "
        "enjoy. Try to understand what went wrong so you can do better next "
        "time. Reach out to someone you trust if you need support."
    )
    assert detect_glued_empathy_morphs(text) is False
    assert detect_empathy_collapse(text) is False


def test_full_prompt_plus_reply_with_scripted_history_not_collapsed():
    """Reproduces the exact false positive: collapse detection run on the
    *full* decoded sequence (chat template + scripted transcript history +
    a coherent generated reply), as `generate_text` used to hand callers
    before `stats["new_text"]` was introduced. The scripted history alone
    ("...feel awful...", "...here with you...") plus a normal empathetic
    reply pushed raw word-frequency hits over the old threshold even though
    nothing resembling real repetition collapse occurred.
    """
    full_text = (
        "<|start_header_id|>user<|end_header_id|>\n"
        "I failed my exam today and I feel awful.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "That sounds really hard. I am here with you.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "I dont think I can recover from this.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "What should I do next?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "It is normal to feel that way after a tough test. Allow yourself "
        "to grieve: give yourself permission to process your emotions. It "
        "is okay to feel sad, frustrated, or disappointed. Take care of "
        "yourself: make sure you are eating, sleeping, and doing things "
        "you enjoy. Try to understand what went wrong so you can do "
        "better next time. Reach out to someone you trust if you need "
        "support."
    )
    assert detect_empathy_collapse(full_text) is False
    assert collapse_score(full_text) < 0.5


def test_glued_morph_fallback_still_fires_on_dominated_short_text():
    """The coverage-ratio replacement must still catch genuinely degenerate
    short outputs saturated with empathy morphemes even when a single-letter
    filler separates each occurrence, which breaks the direct
    `_EMPATHY_MORPH_RE` adjacency match and isolates the fallback branch."""
    text = "x feel x sorry x care x under x here " * 3
    assert detect_glued_empathy_morphs(text) is True
    assert detect_empathy_collapse(text) is True
