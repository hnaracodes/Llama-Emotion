"""User-affect perception via AffectEncoder (v1 default; TRIBEv2 optional elsewhere)."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from src.affective.encoder import AffectEncoder
from src.affective.pipeline import run_encoder_pipeline


def estimate_user_affect(
    messages: Sequence[Any],
    *,
    encoder: AffectEncoder | None = None,
) -> Tuple[np.ndarray, str]:
    """
    Per-turn user affect in 32-d (last timestep summary).

    Returns:
        vector: (32,) float32
        source: e.g. encoder:empatheticdialogues_v1
    """
    if encoder is None:
        encoder = AffectEncoder()

    pipe, source = run_encoder_pipeline(
        messages,
        encoder=encoder,
        source_prefix="encoder:empatheticdialogues_v1",
    )
    vec = pipe["affective_vectors"][-1].astype(np.float32)
    return vec, source
