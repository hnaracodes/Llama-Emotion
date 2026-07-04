"""
Phase 4 extended: strength sweep, logit KL, multi-prompt ablation, hooks on/off.

Usage:
  modal run benchmark_phase4_extended.py
  modal run benchmark_phase4_extended.py --strengths 0,1,2,4
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import (
    ARTIFACTS_MOUNT,
    BENCHMARK_MAX_NEW_TOKENS,
    MODEL_ID,
    PHASE4_ABLATION_PROMPTS,
    PHASE4_STRENGTH_SWEEP,
)


@app.function(image=affective_image, **gpu_kwargs())
def benchmark_phase4_extended(
    strengths: str | None = None,
    skip_strength_sweep: bool = False,
) -> dict:
    from src.benchmark.affect_metrics import compare_generations
    from src.benchmark.phenotype import build_phenotype
    from src.benchmark.hybrid_runner import (
        build_affect_vectors,
        generate_with_affect,
        make_gate,
        prompt_logit_kl_with_affect,
    )
    from src.llm.loader import load_quantized_llama

    strength_list = PHASE4_STRENGTH_SWEEP
    if strengths:
        strength_list = [float(s.strip()) for s in strengths.split(",") if s.strip()]

    model, tokenizer = load_quantized_llama()
    device = next(model.parameters()).device
    gate, _ = make_gate(model)
    vectors = build_affect_vectors(device)

    results: dict = {
        "model_id": MODEL_ID,
        "phase": "4_extended",
        "pipeline_meta": vectors["pipeline_meta"],
        "strength_sweep": [],
        "prompt_ablation": [],
        "summary": {},
    }

    if not skip_strength_sweep:
        canonical = PHASE4_ABLATION_PROMPTS[0]["prompt"]
        sweep_runs = []
        prev_text = None
        for strength in strength_list:
            text, stats = generate_with_affect(
                model,
                tokenizer,
                canonical,
                affect_vector=vectors["aff_high"],
                gate=gate,
                strength=strength,
                max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
                temperature=0.0,
            )
            kl = {}
            if strength > 0:
                kl = prompt_logit_kl_with_affect(
                    model,
                    tokenizer,
                    canonical,
                    vec_a=vectors["aff_low"],
                    vec_b=vectors["aff_high"],
                    gate=gate,
                    strength=strength,
                    hooks_off_a=True,
                )
            entry = {
                "strength": strength,
                "generated_preview": text[-400:],
                "affect_norm": vectors["aff_high_norm"],
                **stats,
                **kl,
            }
            if prev_text is not None:
                entry["text_delta_from_prev_strength"] = compare_generations(
                    prev_text, text, model=model, tokenizer=tokenizer
                )
            sweep_runs.append(entry)
            prev_text = text
        results["strength_sweep"] = sweep_runs

    ablation_strength = 1.0
    prompts_changed = 0
    ablation_rows = []

    for item in PHASE4_ABLATION_PROMPTS:
        prompt_id = item["id"]
        prompt = item["prompt"]

        neutral_text, neutral_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=vectors["aff_high"],
            gate=gate,
            strength=ablation_strength,
            max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
            temperature=0.0,
            hooks_enabled=False,
        )
        high_text, high_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=vectors["aff_high"],
            gate=gate,
            strength=ablation_strength,
            max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
            temperature=0.0,
        )
        hooks_off_text, hooks_off_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=vectors["aff_high"],
            gate=gate,
            strength=ablation_strength,
            max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
            temperature=0.0,
            hooks_enabled=False,
        )

        kl = prompt_logit_kl_with_affect(
            model,
            tokenizer,
            prompt,
            vec_a=vectors["aff_low"],
            vec_b=vectors["aff_high"],
            gate=gate,
            strength=ablation_strength,
            hooks_off_a=True,
        )
        cmp = compare_generations(
            neutral_text, high_text, model=model, tokenizer=tokenizer
        )
        if cmp["text_changed"]:
            prompts_changed += 1

        ablation_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "conditions": {
                    "neutral": {
                        "generated_text": neutral_text,
                        "generated_preview": neutral_text[-400:],
                        **neutral_stats,
                    },
                    "high_affect": {
                        "generated_text": high_text,
                        "generated_preview": high_text[-400:],
                        **high_stats,
                    },
                    "hooks_off_high_vector": {
                        "generated_preview": hooks_off_text[-400:],
                        **hooks_off_stats,
                    },
                },
                "logit_kl": kl,
                "neutral_vs_high_affect": cmp,
                "hooks_effect": compare_generations(
                    hooks_off_text, high_text, model=model, tokenizer=tokenizer
                ),
            }
        )

    results["prompt_ablation"] = ablation_rows
    results["phenotype"] = build_phenotype(ablation_rows)
    results["summary"] = {
        "prompts_total": len(PHASE4_ABLATION_PROMPTS),
        "prompts_text_changed_neutral_vs_high": prompts_changed,
        "fraction_text_changed": round(
            prompts_changed / max(1, len(PHASE4_ABLATION_PROMPTS)), 3
        ),
        "mean_logit_kl_neutral_to_high": round(
            sum(r["logit_kl"]["logit_kl_a_to_b"] for r in ablation_rows)
            / max(1, len(ablation_rows)),
            6,
        ),
        "mean_embedding_cosine_distance": round(
            sum(
                r["neutral_vs_high_affect"].get("embedding_cosine_distance", 0.0)
                for r in ablation_rows
            )
            / max(1, len(ablation_rows)),
            6,
        ),
    }

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase4_extended.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_volume.commit()
    return results


@app.local_entrypoint()
def main(strengths: str | None = None, skip_strength_sweep: bool = False):
    result = benchmark_phase4_extended.remote(
        strengths=strengths,
        skip_strength_sweep=skip_strength_sweep,
    )
    print(json.dumps(result, indent=2))
