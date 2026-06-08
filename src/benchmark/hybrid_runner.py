"""Shared hybrid inference helpers for affect benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.config import AFFECT_DIM, DELTA_THETA
from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks
from src.llm.loader import generate_text


def build_affect_vectors(device: torch.device) -> dict[str, Any]:
    """
    Synthetic TRIBEv2 path → SNN → neutral (zeros) and high_affect (amplified) vectors.
    """
    from src.affective.tribev2_client import pipeline_to_spikes, synthetic_fmri_timeseries
    from src.brain.lif_network import LIFAmygdala, sequence_affective_vectors

    fmri = synthetic_fmri_timeseries(n_timesteps=64, n_vertices=512)
    pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32)).to(device)

    amygdala = LIFAmygdala(input_dim=pipe["D"], output_dim=AFFECT_DIM).to(device)
    aff_seq = sequence_affective_vectors(spikes, amygdala)
    aff_high = torch.from_numpy(aff_seq[-1]).to(device=device, dtype=torch.float32) * 2.0
    aff_low = torch.zeros(AFFECT_DIM, device=device, dtype=torch.float32)

    return {
        "aff_low": aff_low,
        "aff_high": aff_high,
        "aff_high_norm": float(torch.linalg.norm(aff_high).item()),
        "pipeline_meta": {"T": pipe["T"], "D": pipe["D"], "spike_sparsity": pipe["spike_sparsity"]},
    }


def make_gate(model) -> AffectiveGate:
    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device
    return AffectiveGate(AFFECT_DIM, hidden_size, mode="additive").to(device)


@torch.inference_mode()
def generate_with_affect(
    model,
    tokenizer,
    prompt: str,
    *,
    affect_vector: torch.Tensor,
    gate: AffectiveGate,
    strength: float = 1.0,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    hooks_enabled: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Generate text under a fixed affect vector and optional hook strength."""
    device = next(model.parameters()).device
    state = AffectiveState(AFFECT_DIM, device=str(device))
    state.set(affect_vector.to(device=device, dtype=torch.float32))

    handles = []
    if hooks_enabled and strength > 0:
        handles = register_affective_hooks(
            model, gate, state.get, strength=strength
        )

    try:
        text, stats = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    finally:
        for h in handles:
            h.remove()

    return text, stats


@torch.inference_mode()
def prompt_logit_kl_with_affect(
    model,
    tokenizer,
    prompt: str,
    *,
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    gate: AffectiveGate,
    strength: float = 1.0,
) -> dict[str, float]:
    """KL divergence at last prompt token between two affect conditions."""
    from src.benchmark.affect_metrics import kl_divergence_from_logits, last_token_logits

    device = next(model.parameters()).device
    state = AffectiveState(AFFECT_DIM, device=str(device))

    def run_with_vec(vec: torch.Tensor) -> torch.Tensor:
        state.set(vec.to(device=device, dtype=torch.float32))
        handles = register_affective_hooks(model, gate, state.get, strength=strength)
        try:
            return last_token_logits(model, tokenizer, prompt)
        finally:
            for h in handles:
                h.remove()

    logits_a = run_with_vec(vec_a)
    logits_b = run_with_vec(vec_b)
    return {
        "logit_kl_a_to_b": kl_divergence_from_logits(logits_a, logits_b),
        "logit_kl_b_to_a": kl_divergence_from_logits(logits_b, logits_a),
    }
