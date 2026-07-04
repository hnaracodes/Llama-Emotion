"""
Chat soak benchmark (Phase 3E, docs/chat_hardening_plan.md): drive the real
interactive `ChatEngine` — not the one-shot `generate_with_affect` helper used
by other benchmarks — through a long multi-turn distress arc at the actual
chat generation length (`CHAT_MAX_NEW_TOKENS=256`, 4x the 64-token length used
in most other benchmarks and >2x the 96-token gate training holdout length).

This exercises the exact code path a real chat session takes, including the
Phase 1A runtime collapse guard, so a pass here is direct evidence that
`chat.py` / the Modal `EmotionalChatWorker` are safe for real multi-turn use
— not just the isolated single-turn benchmarks.

Usage:
  modal run benchmark_phase_chat_soak.py
  modal run benchmark_phase_chat_soak.py --fail-on-collapse=False
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import ARTIFACTS_MOUNT, CHAT_MAX_NEW_TOKENS, MODEL_ID

# Chained from two already-holdout distress-recovery scenario scripts (only
# the *user* turns — the model generates its own assistant replies, unlike
# the scripted-transcript benchmarks). >= 10 turns to soak-test long,
# stateful sessions per docs/chat_hardening_plan.md Phase 3.
_SOAK_SCENARIO_IDS = ["distress_panic_easing", "distress_job_loss_recovery"]

# 10 turns x 256 tokens plus encoder/SNN overhead needs more headroom than
# the 600s default GPU_TIMEOUT_SEC used by single-shot benchmarks.
_soak_gpu_kwargs = {**gpu_kwargs(), "timeout": 1800}


@app.function(image=affective_image, **_soak_gpu_kwargs)
def benchmark_phase_chat_soak(
    max_new_tokens: int = CHAT_MAX_NEW_TOKENS,
    fail_on_collapse: bool = True,
) -> dict:
    import numpy as np

    from src.benchmark.affect_metrics import empathy_lexical_score
    from src.benchmark.scenarios import load_holdout_scenarios
    from src.chat.engine import ChatEngine
    from src.llm.loader import load_quantized_llama

    all_scenarios = {s.id: s for s in load_holdout_scenarios()}
    user_turns: list[str] = []
    for sid in _SOAK_SCENARIO_IDS:
        sc = all_scenarios.get(sid)
        if sc is None:
            continue
        user_turns.extend(
            m["content"] for m in sc.transcript_messages() if m["role"] == "user"
        )
    if not user_turns:
        raise RuntimeError(
            f"None of {_SOAK_SCENARIO_IDS} found in data/scenarios/ — cannot soak test."
        )

    model, tokenizer = load_quantized_llama()
    engine = ChatEngine(model, tokenizer)
    gate_health = engine.gate_health()

    turns: list[dict] = []
    try:
        for i, user_text in enumerate(user_turns):
            result = engine.generate_reply(user_text, max_new_tokens=max_new_tokens, temperature=0.0)
            metrics = engine.session.turn_metrics[-1]
            turns.append(
                {
                    "turn": i,
                    "user_text": user_text,
                    "reply_preview": result["reply"][-300:],
                    "dominant_tone": result["dominant_tone"],
                    "empathy_lexical_score": round(
                        empathy_lexical_score(result["reply"]), 4
                    ),
                    **metrics,
                }
            )
    finally:
        engine.cleanup()

    collapsed_unrecovered = [
        t for t in turns if t["collapse_detected"] and not t["recovered"]
    ]
    recovered_turns = [t for t in turns if t["recovered"]]
    affect_norms = [t["affect_vector_norm"] for t in turns]

    summary = {
        "turns_run": len(turns),
        "scenario_ids": _SOAK_SCENARIO_IDS,
        "max_new_tokens": max_new_tokens,
        "gate_health": gate_health,
        "collapse_count": sum(1 for t in turns if t["collapse_detected"]),
        "collapse_unrecovered_count": len(collapsed_unrecovered),
        "recovered_count": len(recovered_turns),
        "passed": len(collapsed_unrecovered) == 0,
        "mean_empathy_lexical_score": round(
            float(np.mean([t["empathy_lexical_score"] for t in turns])), 4
        ),
        "affect_norm_trajectory": affect_norms,
        "affect_norm_drift": round(
            float(affect_norms[-1] - affect_norms[0]) if len(affect_norms) > 1 else 0.0,
            4,
        ),
    }

    result = {
        "model_id": MODEL_ID,
        "phase": "chat_soak",
        "turns": turns,
        "summary": summary,
    }

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase_chat_soak.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_volume.commit()

    if fail_on_collapse and not summary["passed"]:
        raise RuntimeError(
            f"Chat soak FAILED: {len(collapsed_unrecovered)} unrecovered collapse(s) "
            f"out of {len(turns)} turns — see phase_chat_soak.json for previews."
        )

    return result


@app.local_entrypoint()
def main(max_new_tokens: int = CHAT_MAX_NEW_TOKENS, fail_on_collapse: bool = True):
    print(
        json.dumps(
            benchmark_phase_chat_soak.remote(max_new_tokens, fail_on_collapse), indent=2
        )
    )
