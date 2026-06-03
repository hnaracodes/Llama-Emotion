"""
Phase 4: Hybrid inference — 4-bit Llama + affective hooks.

Usage:
  modal run run_hybrid.py
  modal run run_hybrid.py --strength 2.0
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np
import torch

from src.common import app, gpu_kwargs, image, model_volume
from src.config import (
    AFFECT_DIM,
    ARTIFACTS_MOUNT,
    BENCHMARK_MAX_NEW_TOKENS,
    BENCHMARK_PROMPT,
    DELTA_THETA,
    MODEL_ID,
)


@app.function(image=image, **gpu_kwargs())
def run_hybrid_inference(strength: float = 1.0) -> dict:
    from src.affective.tribev2_client import pipeline_to_spikes, synthetic_fmri_timeseries
    from src.brain.lif_network import LIFAmygdala, sequence_affective_vectors
    from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks
    from src.llm.loader import generate_text, load_quantized_llama

    # Affective pipeline (synthetic on Modal unless TRIBEv2 installed)
    fmri = synthetic_fmri_timeseries(n_timesteps=64, n_vertices=512)
    pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32))

    amygdala = LIFAmygdala(input_dim=pipe["D"], output_dim=AFFECT_DIM)
    model, tokenizer = load_quantized_llama()
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    gate = AffectiveGate(AFFECT_DIM, hidden_size, mode="additive").to(device)
    amygdala = amygdala.to(device)
    aff_seq = sequence_affective_vectors(spikes.to(device), amygdala)
    aff_high = torch.from_numpy(aff_seq[-1]).to(device=device, dtype=torch.float32) * 2.0
    aff_low = torch.zeros(AFFECT_DIM, device=device)
    state = AffectiveState(AFFECT_DIM, device=str(device))

    results = {"model_id": MODEL_ID, "prompt": BENCHMARK_PROMPT, "conditions": {}}

    for name, vec in [("neutral", aff_low), ("high_affect", aff_high)]:
        state.set(vec)
        handles = register_affective_hooks(
            model, gate, state.get, strength=strength
        )
        try:
            text, stats = generate_text(
                model,
                tokenizer,
                BENCHMARK_PROMPT,
                max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
                temperature=0.0,
            )
            results["conditions"][name] = {
                "generated_preview": text[-400:],
                **stats,
            }
        finally:
            for h in handles:
                h.remove()

    # KL proxy: compare logits on last token would need extra forward — store previews
    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase4_hybrid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_volume.commit()
    return results


@app.local_entrypoint()
def main(strength: float = 1.0):
    result = run_hybrid_inference.remote(strength=strength)
    print(json.dumps(result, indent=2))
