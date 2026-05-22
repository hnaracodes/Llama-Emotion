"""Unit tests for LIF amygdala."""

import numpy as np
import torch

from src.brain.lif_network import LIFAmygdala, run_amygdala_on_spikes


def test_lif_forward_shape():
    spikes = torch.rand(30, 32)
    spikes = (spikes > 0.9).float()
    model = LIFAmygdala(input_dim=32, output_dim=32)
    aff, stats = run_amygdala_on_spikes(spikes, model)
    assert aff.shape == (32,)
    assert not np.isnan(aff).any()
    assert "mean_firing_rate" in stats
