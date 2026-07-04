"""Encoder-based affect pipeline (v1 default path — replaces TRIBEv2 for training/inference)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from src.affective.compress import normalize_affective
from src.affective.emotion_lexicon import emotion_to_32d
from src.affective.encoder import AffectEncoder
from src.config import AFFECT_DIM, DELTA_THETA
from src.encoder.delta_mod import delta_modulate, spike_sparsity


def run_encoder_pipeline(
    messages: Sequence[Any],
    *,
    encoder: AffectEncoder | None = None,
    theta: float = DELTA_THETA,
    source_prefix: str = "encoder:empatheticdialogues_v1",
) -> tuple[dict[str, Any], str]:
    """
    Transcript → 32-d trajectory → normalized vectors → delta-mod spikes.

    Returns:
        pipeline dict (affective_vectors, spikes, T, D, spike_sparsity)
        source string for artifacts/UI
    """
    if encoder is None:
        encoder = AffectEncoder(output_dim=AFFECT_DIM)

    traj = encoder.encode_transcript(messages)
    if traj.size == 0:
        traj = emotion_to_32d("neutral").reshape(1, -1)

    affective = normalize_affective(traj)
    spikes = delta_modulate(affective, theta=theta)
    spike_arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)

    pipe = {
        "affective_vectors": affective,
        "spikes": spike_arr.astype(np.float32),
        "spike_sparsity": spike_sparsity(spikes),
        "T": int(affective.shape[0]),
        "D": int(affective.shape[1]),
    }
    source = source_prefix
    return pipe, source
