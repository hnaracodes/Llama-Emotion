"""Unit tests for delta-modulation encoder."""

import numpy as np
import torch

from src.encoder.delta_mod import delta_modulate, spike_sparsity


def test_delta_mod_step_change():
    # Step at t=1 should spike
    signal = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    spikes = delta_modulate(signal, theta=0.5)
    assert spikes.shape == (5, 1)
    assert spikes[1, 0] == 1.0
    assert spikes[0, 0] == 0.0


def test_delta_mod_sine_sparse():
    t = np.linspace(0, 2 * np.pi, 50)
    signal = np.sin(t)[:, None].astype(np.float32) * 0.01
    spikes = delta_modulate(signal, theta=0.05)
    sp = spike_sparsity(spikes)
    assert 0.0 <= sp <= 1.0
    assert spikes.dtype == torch.float32


def test_short_signal():
    spikes = delta_modulate(np.array([1.0, 2.0]), theta=0.1)
    assert spikes.shape[0] == 2
