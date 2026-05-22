"""LIF spiking amygdala — maps spikes to 32-d affective state."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from src.config import AFFECT_DIM, SNN_BETA, SNN_HIDDEN, SNN_THRESHOLD


class LIFAmygdala(nn.Module):
    """
    Feedforward SNN: spike input (T, F) → hidden LIF → output LIF.
    Exposes firing rates and membrane summaries for affective_vector.
    """

    def __init__(
        self,
        input_dim: int = AFFECT_DIM,
        hidden_dim: int = SNN_HIDDEN,
        output_dim: int = AFFECT_DIM,
        beta: float = SNN_BETA,
        threshold: float = SNN_THRESHOLD,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.output_dim = output_dim

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            spikes: (T, F) or (B, T, F) — if 2D, batch dim added internally

        Returns:
            affective_vector: (D,) last-timestep summary
            stats: firing rates, mem states
        """
        if spikes.dim() == 2:
            spikes = spikes.unsqueeze(0)
        # snntorch expects (B, T, F) for sequential mode
        x = spikes.float()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_hist = []
        mem2_hist = []

        for t in range(x.shape[1]):
            cur = x[:, t, :]
            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            spk2_hist.append(spk2)
            mem2_hist.append(mem2)

        spk2_stack = torch.stack(spk2_hist, dim=1)  # (B, T, D)
        mem2_stack = torch.stack(mem2_hist, dim=1)
        firing_rate = spk2_stack.mean(dim=1)  # (B, D)
        mem_summary = mem2_stack[:, -1, :]
        # Combine rate + membrane → affective vector (trim/pad to output_dim)
        aff = 0.5 * firing_rate + 0.5 * mem_summary
        aff = aff.squeeze(0)

        stats = {
            "mean_firing_rate": float(spk2_stack.mean().item()),
            "firing_rate_per_dim": firing_rate.squeeze(0).detach().cpu().numpy(),
        }
        return aff, stats


def run_amygdala_on_spikes(
    spikes: np.ndarray | torch.Tensor,
    model: LIFAmygdala | None = None,
) -> Tuple[np.ndarray, dict]:
    """Run SNN on spike array (T, F)."""
    if model is None:
        model = LIFAmygdala(input_dim=spikes.shape[-1] if spikes.ndim > 1 else AFFECT_DIM)
    if isinstance(spikes, np.ndarray):
        spikes = torch.from_numpy(spikes.astype(np.float32))
    model.eval()
    with torch.no_grad():
        aff, stats = model(spikes)
    return aff.cpu().numpy().astype(np.float32), stats


def sequence_affective_vectors(
    spikes: torch.Tensor,
    model: LIFAmygdala,
    window: int = 8,
) -> np.ndarray:
    """Sliding window: one 32-d vector per timestep for hybrid sync."""
    T = spikes.shape[0]
    out = np.zeros((T, model.output_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for t in range(T):
            start = max(0, t - window + 1)
            chunk = spikes[start : t + 1]
            if chunk.shape[0] < 2:
                chunk = torch.cat(
                    [torch.zeros(1, spikes.shape[1]), chunk], dim=0
                )
            aff, _ = model(chunk)
            out[t] = aff.numpy()
    return out
