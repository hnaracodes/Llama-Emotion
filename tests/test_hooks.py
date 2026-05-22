"""Unit tests for affective gate projection."""

import torch

from src.llm.hooks import AffectiveGate


def test_affective_gate_additive():
    gate = AffectiveGate(32, 128, mode="additive")
    aff = torch.randn(32)
    mod = gate(aff)
    assert mod.shape[-1] == 128


def test_affective_gate_scale():
    gate = AffectiveGate(32, 128, mode="scale")
    mod = gate(torch.zeros(32))
    assert mod.shape == (1, 128) or mod.shape == (128,)
