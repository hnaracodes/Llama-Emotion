"""Delta-modulation encoder: continuous signals → sparse binary spikes."""

from __future__ import annotations

import numpy as np
import torch


def delta_modulate(
    signal: np.ndarray | torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """
    Emit spike (1) when |x[t] - x[t-1]| > theta per feature.

    Args:
        signal: shape (T, F) or (T,)
        theta: change threshold

    Returns:
        spikes: float tensor shape (T, F), values in {0.0, 1.0}
    """
    if isinstance(signal, np.ndarray):
        x = torch.from_numpy(signal.astype(np.float32))
    else:
        x = signal.float()

    if x.dim() == 1:
        x = x.unsqueeze(-1)

    if x.shape[0] < 2:
        return torch.zeros_like(x)

    diff = torch.abs(x[1:] - x[:-1])
    spikes = (diff > theta).float()
    # Align length T: prepend no-spike row for t=0
    pad = torch.zeros(1, spikes.shape[1], dtype=spikes.dtype)
    return torch.cat([pad, spikes], dim=0)


def spike_sparsity(spikes: torch.Tensor) -> float:
    """Fraction of timesteps with any spike."""
    if spikes.numel() == 0:
        return 0.0
    per_t = spikes.sum(dim=-1) > 0
    return float(per_t.float().mean().item())
