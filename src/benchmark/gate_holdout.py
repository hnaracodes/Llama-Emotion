"""Holdout eval helpers for gate training early-stop and collapse detection."""

from __future__ import annotations

import re
from typing import Any

from src.config import GATE_TRAIN_HOLDOUT_PROMPTS

_EMPATHY_REPEAT_RE = re.compile(
    r"(sorry|feel|care|help|understand)\1{3,}|"
    r"(sorry|feel|care|help|understand)(?:\1){3,}",
    re.IGNORECASE,
)

_EMPATHY_MORPH_RE = re.compile(
    r"(?:here|under|stand|care|feel|sorry|help|support){4,}",
    re.IGNORECASE,
)


def max_empathy_token_run_length(text: str) -> int:
    """Longest consecutive run of empathy-related tokens (word-level)."""
    terms = {"sorry", "feel", "care", "help", "understand"}
    words = re.findall(r"[a-z]+", text.lower())
    best = 0
    run = 0
    prev: str | None = None
    for w in words:
        if w in terms:
            run = run + 1 if w == prev else 1
            prev = w
            best = max(best, run)
        else:
            run = 0
            prev = None
    return best


def detect_glued_empathy_morphs(text: str) -> bool:
    """Detect mashed empathy morpheme chains (hereunderstand, carecare, …)."""
    compact = re.sub(r"[^a-z]", "", text.lower())
    if len(compact) < 12:
        return False
    if _EMPATHY_MORPH_RE.search(compact):
        return True
    glued = (
        "hereunderstand",
        "understandhere",
        "carehere",
        "feelsorry",
        "sorryfeel",
        "sorrysorry",
        "carecare",
    )
    for chunk in glued:
        if compact.count(chunk) >= 2:
            return True
    if len(compact) >= 24:
        morphemes = ("sorry", "feel", "care", "under", "here")
        hits = sum(compact.count(s) for s in morphemes)
        # Character-set diversity is bounded by the alphabet size (<=26), so
        # for any text longer than ~100 chars `len(set(compact))` is *always*
        # <= `len(compact) // 4` — the old check degenerated into a bare
        # `hits >= 6`, which false-positives on long, coherent text that
        # simply mentions "feel"/"care"/"here" a handful of times across
        # separate sentences (e.g. a genuinely empathetic multi-sentence
        # reply). Use a length-normalized coverage ratio instead so the
        # check only fires when these morphemes actually dominate the text,
        # which is what "glued repetition collapse" looks like.
        covered_chars = sum(len(s) * compact.count(s) for s in morphemes)
        coverage = covered_chars / len(compact)
        if hits >= 6 and coverage >= 0.35:
            return True
    return False


def collapse_score(text: str, *, max_run: int | None = None) -> float:
    """Composite collapse score in [0, 1]; higher = worse."""
    from src.config import GATE_COLLAPSE_MAX_RUN

    limit = GATE_COLLAPSE_MAX_RUN if max_run is None else max_run
    score = 0.0
    if _EMPATHY_REPEAT_RE.search(text):
        score += 0.55
    if detect_glued_empathy_morphs(text):
        score += 0.45
    run = max_empathy_token_run_length(text)
    if run > limit:
        score += min(0.35, (run - limit) / 15.0)
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            score += 0.25
    return min(1.0, score)


def detect_empathy_collapse(text: str, *, max_run: int | None = None) -> bool:
    """True when generation shows empathy-token repetition collapse."""
    from src.config import GATE_COLLAPSE_MAX_RUN

    limit = GATE_COLLAPSE_MAX_RUN if max_run is None else max_run
    if _EMPATHY_REPEAT_RE.search(text):
        return True
    if detect_glued_empathy_morphs(text):
        return True
    if max_empathy_token_run_length(text) > limit:
        return True
    return collapse_score(text, max_run=max_run) >= 0.5


def holdout_prompts() -> list[dict[str, str]]:
    """Fixed prompts for gate training-time checkpoint selection.

    Deliberately disjoint from PHASE4_ABLATION_PROMPTS (never used for gate
    training or checkpoint selection) so Phase 4 ablation / chat A-B results
    remain an independent check of the saved checkpoint.
    """
    return list(GATE_TRAIN_HOLDOUT_PROMPTS)


def summarize_holdout_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    collapsed = sum(1 for r in rows if r.get("collapse_detected"))
    scores = [float(r.get("collapse_score", 0.0)) for r in rows]
    return {
        "prompts": len(rows),
        "collapse_count": collapsed,
        "any_collapse": collapsed > 0,
        "max_collapse_score": max(scores) if scores else 0.0,
        "mean_collapse_score": sum(scores) / len(scores) if scores else 0.0,
        "rows": rows,
    }
