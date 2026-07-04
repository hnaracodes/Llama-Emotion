"""Build behavioral phenotype card from ablation rows (Track C)."""

from __future__ import annotations

from typing import Any

from src.benchmark.affect_metrics import empathy_lexical_score, sentiment_lexical_score


def _hedging_rate(text: str) -> float:
    terms = ("maybe", "perhaps", "might", "could", "possibly", "somewhat")
    words = max(1, len(text.split()))
    return sum(text.lower().count(t) for t in terms) / words


def _question_rate(text: str) -> float:
    return text.count("?") / max(1, len(text.split()))


def build_phenotype(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-prompt neutral vs high-affect deltas.

    Each row should include at least:
      - affect: str label (e.g. neutral, high)
      - text: generated string
    Or a nested structure from phase4 ablation with conditions.neutral/high_affect.
    """
    if not prompt_results:
        return {
            "metric_type": "heuristic",
            "empathy_delta": 0.0,
            "sentiment_delta": 0.0,
            "verbosity_delta": 0.0,
            "hedging_delta": 0.0,
            "question_delta": 0.0,
            "n_prompts": 0,
        }

    empathy_neutral: list[float] = []
    empathy_high: list[float] = []
    sentiment_neutral: list[float] = []
    sentiment_high: list[float] = []
    len_neutral: list[int] = []
    len_high: list[int] = []
    hedge_neutral: list[float] = []
    hedge_high: list[float] = []
    q_neutral: list[float] = []
    q_high: list[float] = []

    for row in prompt_results:
        if "conditions" in row:
            neutral_text = (
                row["conditions"]["neutral"].get("generated_text")
                or row["conditions"]["neutral"].get("generated_preview", "")
            )
            high_text = (
                row["conditions"]["high_affect"].get("generated_text")
                or row["conditions"]["high_affect"].get("generated_preview", "")
            )
        else:
            if row.get("affect") == "neutral":
                neutral_text = row.get("text", "")
                high_text = ""
            else:
                neutral_text = ""
                high_text = row.get("text", "")
        if not neutral_text and not high_text:
            continue
        if neutral_text:
            empathy_neutral.append(empathy_lexical_score(neutral_text))
            sentiment_neutral.append(sentiment_lexical_score(neutral_text))
            len_neutral.append(len(neutral_text.split()))
            hedge_neutral.append(_hedging_rate(neutral_text))
            q_neutral.append(_question_rate(neutral_text))
        if high_text:
            empathy_high.append(empathy_lexical_score(high_text))
            sentiment_high.append(sentiment_lexical_score(high_text))
            len_high.append(len(high_text.split()))
            hedge_high.append(_hedging_rate(high_text))
            q_high.append(_question_rate(high_text))

    def _delta(high: list[float], low: list[float]) -> float:
        if not high or not low:
            return 0.0
        return round(float(sum(high) / len(high) - sum(low) / len(low)), 4)

    return {
        "metric_type": "heuristic",
        "empathy_delta": _delta(empathy_high, empathy_neutral),
        "sentiment_delta": _delta(sentiment_high, sentiment_neutral),
        "verbosity_delta": _delta(
            [float(x) for x in len_high], [float(x) for x in len_neutral]
        ),
        "hedging_delta": _delta(hedge_high, hedge_neutral),
        "question_delta": _delta(q_high, q_neutral),
        "n_prompts": max(len(empathy_high), len(empathy_neutral)),
    }
