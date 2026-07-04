"""Affect vector normalization utilities."""

from __future__ import annotations

import numpy as np
import torch

from src.config import GATE_MAX_AFFECT_NORM


def clip_affect_norm(
    vector: np.ndarray | torch.Tensor,
    *,
    max_norm: float = GATE_MAX_AFFECT_NORM,
) -> np.ndarray | torch.Tensor:
    """Scale vector down if L2 norm exceeds max_norm."""
    if isinstance(vector, torch.Tensor):
        norm = float(torch.linalg.norm(vector).item())
        if norm <= max_norm or norm < 1e-8:
            return vector
        return vector * (max_norm / norm)
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= max_norm or norm < 1e-8:
        return arr
    return (arr * (max_norm / norm)).astype(np.float32)
