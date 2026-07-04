"""
Holdout scenario eval: hooks-off vs transcript-conditioned affect (Track C / M5).

Runs encoder→SNN on synthetic multi-turn scripts; compares generation under
hooks-off (AF-4 neutral) vs hooks-on with scenario-derived affect.

Usage:
  modal run benchmark_phase_scenarios.py
  modal run benchmark_phase_scenarios.py --scenario-ids tone_calm_to_panic,factual_week_planning
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import ARTIFACTS_MOUNT, BENCHMARK_MAX_NEW_TOKENS, CHAT_MAX_NEW_TOKENS, MODEL_ID


@app.function(image=affective_image, **gpu_kwargs())
def benchmark_phase_scenarios(
    scenario_ids: list[str] | None = None,
    max_scenarios: int = 8,
    max_new_tokens: int = BENCHMARK_MAX_NEW_TOKENS,
) -> dict:
    import numpy as np
    import torch

    from src.affective.pipeline import run_encoder_pipeline
    from src.benchmark.affect_metrics import compare_generations, empathy_lexical_score
    from src.benchmark.hybrid_runner import generate_with_affect, make_gate
    from src.benchmark.scenarios import load_holdout_scenarios
    from src.brain.checkpoints import load_amygdala, load_encoder
    from src.chat.session import ChatMessage
    from src.chat.signatures import extract_signature_from_pipeline
    from src.chat.tone_markers import dominant_tone
    from src.chat.transcript import build_llama_prompt
    from src.config import AFFECT_DIM
    from src.llm.loader import load_quantized_llama

    all_scenarios = load_holdout_scenarios()
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in all_scenarios if s.id in wanted]
    else:
        # Representative slice: edge categories + hooks-off A/B tagged factual
        priority = [
            "tone_calm_to_panic",
            "distress_job_loss_recovery",
            "conflict_escalation",
            "factual_week_planning",
            "tone_whiplash_bad_news",
            "factual_recipe_question",
            "distress_panic_easing",
            "tone_joking_then_serious",
        ]
        by_id = {s.id: s for s in all_scenarios}
        scenarios = [by_id[sid] for sid in priority if sid in by_id]
        if len(scenarios) < max_scenarios:
            for s in all_scenarios:
                if s not in scenarios:
                    scenarios.append(s)
                if len(scenarios) >= max_scenarios:
                    break
        scenarios = scenarios[:max_scenarios]

    model, tokenizer = load_quantized_llama()
    device = next(model.parameters()).device
    gate, gate_meta = make_gate(model)
    encoder, enc_meta = load_encoder(device="cpu")
    amygdala, amy_meta = load_amygdala(input_dim=AFFECT_DIM, device=str(device))

    rows: dict[str, dict] = {}
    for sc in scenarios:
        msgs = sc.transcript_messages()
        pipe, source = run_encoder_pipeline(msgs, encoder=encoder)
        fmri_ts = np.zeros((pipe["T"], 1), dtype=np.float32)
        sig = extract_signature_from_pipeline(
            fmri_ts, pipe, amygdala=amygdala, device=device
        )
        affect_vector = sig["vector"]
        tone = dominant_tone(sig["traits"])

        history = [
            ChatMessage(role=m["role"], content=m["content"])  # type: ignore[arg-type]
            for m in msgs
        ]
        history.append(ChatMessage(role="user", content=sc.eval_question))
        prompt = build_llama_prompt(tokenizer, history, add_generation_prompt=True)

        aff_tensor = torch.from_numpy(affect_vector).to(device=device, dtype=torch.float32)
        zero_tensor = torch.zeros(AFFECT_DIM, device=device, dtype=torch.float32)

        hooks_off_text, hooks_off_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=zero_tensor,
            gate=gate,
            strength=1.0,
            hooks_enabled=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        hooks_on_text, hooks_on_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=aff_tensor,
            gate=gate,
            strength=1.0,
            hooks_enabled=True,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        cmp = compare_generations(hooks_off_text, hooks_on_text, model=model, tokenizer=tokenizer)
        from src.benchmark.gate_holdout import detect_empathy_collapse

        # Collapse detection must only see the model's own newly generated
        # continuation — `hooks_on_text` also contains the full prompt and
        # scripted transcript history, which can legitimately contain
        # empathy-related words and would otherwise cause false positives.
        generated_only = hooks_on_stats.get("new_text", hooks_on_text)

        rows[sc.id] = {
            "category": sc.category,
            "tags": list(sc.tags),
            "turns": sc.turn_count,
            "affect_source": source,
            "dominant_tone": tone,
            "affect_vector_norm": round(float(np.linalg.norm(affect_vector)), 4),
            "empathy_hooks_off": round(empathy_lexical_score(hooks_off_text), 4),
            "empathy_hooks_on": round(empathy_lexical_score(hooks_on_text), 4),
            "hooks_off_preview": hooks_off_text[-400:],
            "hooks_on_preview": hooks_on_text[-400:],
            "hooks_off_stats": hooks_off_stats,
            "hooks_on_stats": hooks_on_stats,
            "collapse_detected": detect_empathy_collapse(generated_only),
            **cmp,
        }

    scientific = (
        enc_meta.source == "trained"
        and amy_meta.source == "trained"
        and gate_meta.source == "trained"
        and len(scenarios) > 0
    )

    text_changed = sum(1 for r in rows.values() if r.get("text_changed"))
    mean_empathy_delta = (
        sum(r.get("empathy_delta", 0.0) for r in rows.values()) / max(1, len(rows))
    )

    result = {
        "model_id": MODEL_ID,
        "phase": "scenario_holdout",
        "metric_type": "heuristic",
        "interpretation": (
            "empathy_delta and sentiment_delta are lexical heuristics only; "
            "not ground-truth empathy. scientific=true requires trained encoder, amygdala, and gate."
        ),
        "encoder_source": enc_meta.source,
        "amygdala_source": amy_meta.source,
        "gate_source": gate_meta.source,
        "scientific": scientific,
        "scenarios": rows,
        "summary": {
            "scenarios_run": len(rows),
            "max_new_tokens": max_new_tokens,
            "categories": sorted({r["category"] for r in rows.values()}),
            "text_changed_count": text_changed,
            "fraction_text_changed": round(text_changed / max(1, len(rows)), 4),
            "mean_empathy_delta_hooks_on_vs_off": round(mean_empathy_delta, 4),
            "collapse_count": sum(
                1 for r in rows.values() if r.get("collapse_detected")
            ),
            "tone_shift_scenarios": [
                k for k, v in rows.items() if "sudden_tone_shift" in v.get("tags", [])
            ],
        },
    }

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase_scenarios.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(
    scenario_ids: str = "",
    max_scenarios: int = 8,
    max_new_tokens: int = BENCHMARK_MAX_NEW_TOKENS,
):
    ids = [s.strip() for s in scenario_ids.split(",") if s.strip()] or None
    print(json.dumps(
        benchmark_phase_scenarios.remote(ids, max_scenarios, max_new_tokens), indent=2
    ))
