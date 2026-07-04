"""Local encoder-only eval on holdout scenarios (no GPU / Llama required)."""

from __future__ import annotations

import json
from pathlib import Path

from src.affective.pipeline import run_encoder_pipeline
from src.affective.encoder import AffectEncoder
from src.benchmark.scenarios import load_holdout_scenarios
from src.chat.tone_markers import dominant_tone
from src.config import ARTIFACTS_DIR
from src.chat.signatures import compute_traits


def run_encoder_eval(scenario_ids: list[str] | None = None) -> dict:
    encoder = AffectEncoder(backend="hash")
    scenarios = load_holdout_scenarios()
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s.id in wanted]

    rows = {}
    for sc in scenarios:
        pipe, source = run_encoder_pipeline(sc.transcript_messages(), encoder=encoder)
        traits = compute_traits(
            pipe["affective_vectors"],
            pipe["spikes"],
            {"mean_firing_rate": float(pipe["spikes"].mean())},
        )
        rows[sc.id] = {
            "category": sc.category,
            "tags": list(sc.tags),
            "turns": sc.turn_count,
            "source": source,
            "dominant_tone": dominant_tone(traits),
            "spike_sparsity": pipe["spike_sparsity"],
            "trait_shift": traits.get("shift", 0.0),
        }

    return {
        "backend": "hash",
        "metric_type": "encoder_only_exploratory",
        "scenarios": rows,
        "summary": {
            "count": len(rows),
            "tone_by_category": {
                cat: sorted(
                    {v["dominant_tone"] for v in rows.values() if v["category"] == cat}
                )
                for cat in sorted({v["category"] for v in rows.values()})
            },
        },
    }


def main() -> None:
    highlight = ["tone_calm_to_panic", "factual_week_planning"]
    result = run_encoder_eval(highlight)
    out = ARTIFACTS_DIR / "scenario_encoder_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
