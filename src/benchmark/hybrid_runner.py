"""Shared hybrid inference helpers for affect benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.affective.affect_norm import clip_affect_norm
from src.config import AFFECT_DIM, DEFAULT_REPETITION_PENALTY, DELTA_THETA, MODEL_ID
from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks
from src.llm.loader import generate_text


def build_affect_vectors(device: torch.device) -> dict[str, Any]:
    """
    Trained encoder → spikes → amygdala → high_affect vector.
    Low baseline is zeros (used only when hooks are on; neutral runs use hooks_off).
    """
    from src.affective.pipeline import run_encoder_pipeline
    from src.brain.checkpoints import load_amygdala, load_encoder
    from src.brain.lif_network import sequence_affective_vectors

    encoder, enc_meta = load_encoder(device="cpu")
    amygdala, amy_meta = load_amygdala(input_dim=AFFECT_DIM, device=str(device))
    msgs = [{"role": "user", "content": "I feel anxious and need support."}]
    pipe, pipe_source = run_encoder_pipeline(msgs, encoder=encoder)
    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32)).to(device)

    aff_seq, _, _ = sequence_affective_vectors(spikes, amygdala)
    aff_high = clip_affect_norm(
        torch.from_numpy(aff_seq[-1]).to(device=device, dtype=torch.float32)
    )
    if aff_high.norm() < 1e-6:
        raise RuntimeError(
            "amygdala produced zero affect vector; check encoder/SNN checkpoints "
            f"(encoder={enc_meta.source}, amygdala={amy_meta.source})"
        )
    aff_low = torch.zeros(AFFECT_DIM, device=device, dtype=torch.float32)

    return {
        "aff_low": aff_low,
        "aff_high": aff_high,
        "aff_high_norm": float(torch.linalg.norm(aff_high).item()),
        "pipeline_meta": {
            "T": pipe["T"],
            "D": pipe["D"],
            "spike_sparsity": pipe["spike_sparsity"],
            "encoder_source": enc_meta.source,
            "amygdala_source": amy_meta.source,
            "pipe_source": pipe_source,
        },
    }


def make_gate(model) -> tuple[AffectiveGate, Any]:
    """Load trained gate when available; otherwise random-init (noop at zero)."""
    from src.brain.checkpoints import load_gate

    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device
    gate = AffectiveGate(AFFECT_DIM, hidden_size, mode="additive").to(device)
    meta = load_gate(gate, model_id=MODEL_ID, hidden_size=hidden_size, device=str(device))
    return gate, meta


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
    clip_affect: bool = True,
    repetition_penalty: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate text under a fixed affect vector and optional hook strength."""
    device = next(model.parameters()).device
    vec = affect_vector.to(device=device, dtype=torch.float32)
    if clip_affect:
        vec = clip_affect_norm(vec)
    state = AffectiveState(AFFECT_DIM, device=str(device))
    state.set(vec)

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
            repetition_penalty=repetition_penalty,
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
    hooks_off_a: bool = False,
) -> dict[str, float]:
    """KL divergence at last prompt token between two affect conditions."""
    from src.benchmark.affect_metrics import kl_divergence_from_logits, last_token_logits

    device = next(model.parameters()).device
    state = AffectiveState(AFFECT_DIM, device=str(device))

    def run_with_vec(vec: torch.Tensor, *, hooks_off: bool) -> torch.Tensor:
        state.set(vec.to(device=device, dtype=torch.float32))
        handles = []
        if not hooks_off and strength > 0:
            handles = register_affective_hooks(model, gate, state.get, strength=strength)
        try:
            return last_token_logits(model, tokenizer, prompt)
        finally:
            for h in handles:
                h.remove()

    logits_a = run_with_vec(vec_a, hooks_off=hooks_off_a)
    logits_b = run_with_vec(vec_b, hooks_off=False)
    return {
        "logit_kl_a_to_b": kl_divergence_from_logits(logits_a, logits_b),
        "logit_kl_b_to_a": kl_divergence_from_logits(logits_b, logits_a),
    }
