"""Spike-timing-dependent plasticity for local SNN adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn


class STDPUpdater:
    """
    Pair-based STDP on linear layer weights.
    A_LTP * exp(-dt/tau_plus) for pre-before-post; A_LTD for post-before-pre.
    """

    def __init__(
        self,
        a_ltp: float = 0.01,
        a_ltd: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        self.a_ltp = a_ltp
        self.a_ltd = a_ltd
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

    def update_linear(
        self,
        layer: nn.Linear,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        pre_spikes, post_spikes: (T, N) binary/float spike trains.

        Returns delta W (same shape as layer.weight) for logging.
        """
        T = pre_spikes.shape[0]
        w = layer.weight.data
        delta = torch.zeros_like(w)
        pre_idx = (pre_spikes > 0.5).nonzero(as_tuple=False)
        post_idx = (post_spikes > 0.5).nonzero(as_tuple=False)

        for pi in pre_idx:
            t_pre, i_pre = int(pi[0]), int(pi[1])
            for pj in post_idx:
                t_post, j_post = int(pj[0]), int(pj[1])
                if i_pre >= pre_spikes.shape[1] or j_post >= post_spikes.shape[1]:
                    continue
                dt = t_post - t_pre
                if dt > 0:
                    delta[j_post, i_pre] += self.a_ltp * torch.exp(
                        torch.tensor(-dt / self.tau_plus)
                    )
                elif dt < 0:
                    delta[j_post, i_pre] -= self.a_ltd * torch.exp(
                        torch.tensor(dt / self.tau_minus)
                    )

        layer.weight.data = torch.clamp(w + delta, self.w_min, self.w_max)
        return delta
