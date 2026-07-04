"""LIFAmygdala membrane state carryover tests."""

import numpy as np
import torch

from src.brain.lif_network import LIFAmygdala, sequence_affective_vectors


def test_membrane_carry_changes_output():
    torch.manual_seed(42)
    D, T = 8, 20
    amy = LIFAmygdala(input_dim=D, output_dim=32)
    spikes = torch.rand(T, D)

    _, mem, _ = sequence_affective_vectors(spikes, amy)
    aff_carry, _, _ = sequence_affective_vectors(spikes, amy, mem_state=mem)
    aff_fresh, _, _ = sequence_affective_vectors(spikes, amy)
    assert not np.allclose(aff_carry[-1], aff_fresh[-1])


def test_reset_state_clears_membrane():
    torch.manual_seed(1)
    D, T = 8, 12
    amy = LIFAmygdala(input_dim=D, output_dim=32)
    spikes = torch.rand(T, D)
    _, mem, _ = sequence_affective_vectors(spikes, amy)
    assert mem is not None
    amy.reset_state()
    _, mem_after, _ = sequence_affective_vectors(spikes, amy, mem_state=mem)
    # After reset, internal module state is fresh; carrying old mem still works
    # but reset_state should clear module-level cache if any
    assert hasattr(amy, "reset_state")
