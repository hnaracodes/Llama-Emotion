"""Affective signature extraction and EMA updates from TRIBEv2 pipeline output."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.config import AFFECT_DIM, AFFECT_EMA_ALPHA, DELTA_THETA


def compute_traits(
    affective_vectors: np.ndarray,
    spikes: np.ndarray,
    snn_stats: dict[str, Any] | None = None,
    *,
    prev_vector: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Derive display-only trait scalars from pipeline outputs.

    affective_vectors: (T, D)
    spikes: (T, D)
    """
    if affective_vectors.size == 0:
        return {
            "engagement": 0.0,
            "arousal": 0.0,
            "tension": 0.0,
            "warmth": 0.0,
            "stability": 1.0,
            "shift": 0.0,
        }

    last = affective_vectors[-1]
    spike_rate = float(np.mean(spikes > 0)) if spikes.size else 0.0
    vec_norm = float(np.linalg.norm(last))
    engagement = float(np.clip(spike_rate * 2.0 + vec_norm * 0.5, 0.0, 1.0))

    firing = 0.0
    if snn_stats and "mean_firing_rate" in snn_stats:
        firing = float(np.clip(snn_stats["mean_firing_rate"], 0.0, 1.0))
    arousal = float(np.clip(firing + spike_rate * 0.5, 0.0, 1.0))

    if affective_vectors.shape[0] >= 2:
        diffs = np.linalg.norm(np.diff(affective_vectors, axis=0), axis=1)
        tension = float(np.clip(np.mean(diffs) * 2.0, 0.0, 1.0))
        stability = float(np.clip(1.0 - np.std(affective_vectors, axis=0).mean(), 0.0, 1.0))
    else:
        tension = 0.2
        stability = 0.8

    # Warmth: mean of first third of ROI channels (proxy)
    d = last.shape[0]
    warmth = float(np.clip(np.mean(last[: max(1, d // 3)]) + 0.5, 0.0, 1.0))

    shift = 0.0
    if prev_vector is not None and prev_vector.shape == last.shape:
        shift = float(np.linalg.norm(last - prev_vector))

    return {
        "engagement": round(engagement, 4),
        "arousal": round(arousal, 4),
        "tension": round(tension, 4),
        "warmth": round(warmth, 4),
        "stability": round(stability, 4),
        "shift": round(shift, 4),
    }


def trait_vector(traits: dict[str, float]) -> np.ndarray:
    keys = ("engagement", "arousal", "tension", "warmth", "stability", "shift")
    return np.array([traits.get(k, 0.0) for k in keys], dtype=np.float32)


def trait_shift_magnitude(
    before: dict[str, float], after: dict[str, float]
) -> float:
    return float(np.linalg.norm(trait_vector(after) - trait_vector(before)))


def ema_update(
    current: np.ndarray | None,
    new: np.ndarray,
    alpha: float | None = None,
) -> np.ndarray:
    alpha = AFFECT_EMA_ALPHA if alpha is None else alpha
    new = np.asarray(new, dtype=np.float32).reshape(-1)
    if current is None:
        return new.copy()
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    if current.shape != new.shape:
        return new.copy()
    return ((1.0 - alpha) * current + alpha * new).astype(np.float32)


def extract_signature_from_pipeline(
    fmri_ts: np.ndarray,
    pipe: dict[str, Any],
    *,
    amygdala: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
    prev_vector: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run SNN on spikes and return vector + traits."""
    from src.brain.lif_network import LIFAmygdala, sequence_affective_vectors

    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32)).to(device)
    if amygdala is None:
        amygdala = LIFAmygdala(input_dim=pipe["D"], output_dim=AFFECT_DIM).to(device)
    amygdala.eval()

    with torch.no_grad():
        aff_seq = sequence_affective_vectors(spikes, amygdala)
        _, stats = amygdala(spikes if spikes.dim() == 2 else spikes.squeeze(0))

    vector = aff_seq[-1].astype(np.float32)
    traits = compute_traits(
        pipe["affective_vectors"],
        pipe["spikes"],
        stats,
        prev_vector=prev_vector,
    )
    return {
        "vector": vector,
        "traits": traits,
        "snn_stats": stats,
        "source_meta": {},
    }


def apply_manual_affect_scale(vector: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is None:
        return vector
    return (vector * float(scale)).astype(np.float32)
