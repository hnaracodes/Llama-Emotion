"""
Phase loop benchmark: multi-turn affect arc + coupling correlation (Track B / M4).

Uses holdout scenarios with >= min_turns (default 8) when available; falls back to
CHAT_AB_TRANSCRIPTS with exploratory label when n_deltas < 5.

Usage:
  modal run benchmark_phase_loop.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import ARTIFACTS_MOUNT, CHAT_AB_TRANSCRIPTS


@app.function(image=affective_image, **gpu_kwargs())
def benchmark_phase_loop(min_turns: int = 8) -> dict:
    import numpy as np

    from src.affective.coupling import affect_coupling_corr, couple
    from src.affective.dynamics import AffectDynamics
    from src.affective.pipeline import run_encoder_pipeline
    from src.benchmark.scenarios import load_holdout_scenarios
    from src.brain.checkpoints import load_amygdala, load_encoder
    from src.config import AFFECT_DIM

    encoder, enc_meta = load_encoder(device="cpu")
    amygdala, amy_meta = load_amygdala(input_dim=AFFECT_DIM, device="cpu")

    holdout = load_holdout_scenarios(min_turns=min_turns)
    arc_specs: list[tuple[str, list[dict[str, str]]]] = []
    if holdout:
        for sc in holdout[:4]:
            arc_specs.append((sc.id, sc.transcript_messages()))
    else:
        for arc_id in ("distress", "hopeful"):
            turns = CHAT_AB_TRANSCRIPTS[arc_id]
            arc_specs.append(
                (arc_id, [{"role": t["role"], "content": t["content"]} for t in turns])
            )

    runs: dict[str, dict] = {}

    for arc_id, msgs in arc_specs:
        pipe, source = run_encoder_pipeline(msgs, encoder=encoder)
        dyn = AffectDynamics()
        internal: list[np.ndarray] = []
        user_vecs = list(pipe["affective_vectors"])
        state = None
        for u in user_vecs:
            state = couple(u, state, coupling=0.35)
            internal.append(dyn.step(state))

        n_deltas = max(0, len(user_vecs) - 1)
        coupling_r = affect_coupling_corr(user_vecs, internal)
        scientific = (
            enc_meta.source == "trained"
            and amy_meta.source == "trained"
            and source.startswith("encoder:")
            and "synthetic" not in source
            and n_deltas >= 5
        )

        runs[arc_id] = {
            "turns": len(msgs),
            "n_deltas": n_deltas,
            "source": source,
            "coupling_corr": round(coupling_r, 4),
            "scientific": scientific,
            "exploratory": n_deltas < 5,
            "encoder_source": enc_meta.source,
            "amygdala_source": amy_meta.source,
        }

    result = {
        "phase": "loop",
        "min_turns_filter": min_turns,
        "arcs": runs,
        "summary": {
            "mean_coupling_corr": round(
                float(np.mean([runs[a]["coupling_corr"] for a in runs])), 4
            )
            if runs
            else 0.0,
            "scientific": all(runs[a]["scientific"] for a in runs) if runs else False,
            "exploratory_arcs": sum(1 for a in runs if runs[a]["exploratory"]),
        },
    }
    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase_loop.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(min_turns: int = 8):
    print(json.dumps(benchmark_phase_loop.remote(min_turns), indent=2))
