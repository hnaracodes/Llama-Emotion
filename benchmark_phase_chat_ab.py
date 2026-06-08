"""
Phase chat A/B: transcript-conditioned affect → generation comparison.

Usage:
  modal run benchmark_phase_chat_ab.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from src.common import app, gpu_kwargs, image, model_volume
from src.config import (
    ARTIFACTS_MOUNT,
    CHAT_AB_TRANSCRIPTS,
    CHAT_AB_USER_QUESTION,
    CHAT_MAX_NEW_TOKENS,
    DELTA_THETA,
    MODEL_ID,
)


@app.function(image=image, **gpu_kwargs())
def benchmark_phase_chat_ab() -> dict:
    import numpy as np
    import torch

    from src.affective.tribev2_client import pipeline_to_spikes, run_tribev2_from_transcript
    from src.benchmark.affect_metrics import compare_generations
    from src.benchmark.hybrid_runner import generate_with_affect, make_gate
    from src.chat.session import ChatMessage, ChatSession
    from src.chat.signatures import ema_update, extract_signature_from_pipeline
    from src.chat.tone_markers import dominant_tone
    from src.chat.transcript import build_llama_prompt
    from src.llm.loader import load_quantized_llama

    model, tokenizer = load_quantized_llama()
    device = next(model.parameters()).device
    gate = make_gate(model)

    scenarios: dict[str, dict] = {}
    generation_by_scenario: dict[str, str] = {}

    t0_base = time.time()
    for scenario_id, turns in CHAT_AB_TRANSCRIPTS.items():
        session = ChatSession()
        for i, turn in enumerate(turns):
            session.append(
                turn["role"],  # type: ignore[arg-type]
                turn["content"],
            )
            session.messages[-1].timestamp = t0_base + i * 45.0

        msgs = session.transcript_messages()
        fmri, source = run_tribev2_from_transcript(msgs)
        pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
        sig = extract_signature_from_pipeline(fmri, pipe, device=device)
        affect_vector = ema_update(None, sig["vector"])
        tone = dominant_tone(sig["traits"])

        scenarios[scenario_id] = {
            "transcript_turns": len(turns),
            "tribe_source": source,
            "traits": sig["traits"],
            "dominant_tone": tone,
            "affect_vector_norm": round(float(np.linalg.norm(affect_vector)), 4),
        }

        history = list(session.messages)
        history.append(ChatMessage(role="user", content=CHAT_AB_USER_QUESTION))
        prompt = build_llama_prompt(tokenizer, history, add_generation_prompt=True)

        aff_tensor = torch.from_numpy(affect_vector).to(device=device, dtype=torch.float32)
        reply, stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=aff_tensor,
            gate=gate,
            strength=1.0,
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            temperature=0.0,
        )
        generation_by_scenario[scenario_id] = reply
        scenarios[scenario_id]["generation"] = {
            "reply_preview": reply[-500:],
            "user_question": CHAT_AB_USER_QUESTION,
            **stats,
        }

    # Pairwise comparisons vs neutral baseline
    baseline = "neutral"
    comparisons = []
    if baseline in generation_by_scenario:
        base_text = generation_by_scenario[baseline]
        for scenario_id, text in generation_by_scenario.items():
            if scenario_id == baseline:
                continue
            cmp = compare_generations(
                base_text, text, model=model, tokenizer=tokenizer
            )
            comparisons.append(
                {
                    "baseline": baseline,
                    "scenario": scenario_id,
                    "baseline_tone": scenarios[baseline]["dominant_tone"],
                    "scenario_tone": scenarios[scenario_id]["dominant_tone"],
                    "affect_norm_delta": round(
                        scenarios[scenario_id]["affect_vector_norm"]
                        - scenarios[baseline]["affect_vector_norm"],
                        4,
                    ),
                    **cmp,
                }
            )

    # Distress vs hopeful (mood arc within same opening)
    if "distress" in generation_by_scenario and "hopeful" in generation_by_scenario:
        mood_arc = compare_generations(
            generation_by_scenario["distress"],
            generation_by_scenario["hopeful"],
            model=model,
            tokenizer=tokenizer,
        )
    else:
        mood_arc = {}

    results = {
        "model_id": MODEL_ID,
        "phase": "chat_ab",
        "scenarios": scenarios,
        "comparisons_vs_neutral": comparisons,
        "distress_vs_hopeful": mood_arc,
        "summary": {
            "scenarios_run": len(scenarios),
            "tone_by_scenario": {
                k: v["dominant_tone"] for k, v in scenarios.items()
            },
            "comparisons_text_changed": sum(
                1 for c in comparisons if c.get("text_changed")
            ),
            "mean_empathy_delta_vs_neutral": round(
                sum(c.get("empathy_delta", 0.0) for c in comparisons)
                / max(1, len(comparisons)),
                4,
            ),
            "mean_sentiment_delta_vs_neutral": round(
                sum(c.get("sentiment_delta", 0.0) for c in comparisons)
                / max(1, len(comparisons)),
                4,
            ),
        },
    }

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase_chat_ab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_volume.commit()
    return results


@app.local_entrypoint()
def main():
    result = benchmark_phase_chat_ab.remote()
    print(json.dumps(result, indent=2))
