"""Unit tests for affective gate projection (Track D)."""

import torch
import numpy as np

from src.llm.hooks import AffectiveGate, AffectiveState, make_hidden_state_hook


def test_affective_gate_additive():
    gate = AffectiveGate(32, 128, mode="additive")
    aff = torch.randn(32)
    mod = gate(aff)
    assert mod.shape[-1] == 128


def test_affective_gate_scale():
    gate = AffectiveGate(32, 128, mode="scale")
    mod = gate(torch.zeros(32))
    assert mod.shape == (1, 128) or mod.shape == (128,)


def test_affective_gate_zero_affect_additive_is_bias_only():
    """Zero affect → projection bias only (gate bias init is zero)."""
    gate = AffectiveGate(32, 128, mode="additive")
    mod = gate(torch.zeros(32))
    assert torch.allclose(mod, torch.zeros_like(mod))


def test_hidden_state_hook_zero_affect_is_identity():
    """Additive hook with zero affect leaves hidden states unchanged."""
    gate = AffectiveGate(32, 8, mode="additive")
    hidden = torch.randn(1, 3, 8)
    hook = make_hidden_state_hook(gate, lambda: torch.zeros(32), strength=1.0)

    out = hook(None, None, hidden)
    assert torch.allclose(out, hidden)


def test_hidden_state_hook_none_affect_skips_modulation():
    gate = AffectiveGate(32, 8, mode="additive")
    hidden = torch.randn(1, 3, 8)
    hook = make_hidden_state_hook(gate, lambda: None, strength=1.0)

    out = hook(None, None, hidden)
    assert out is hidden


def test_hidden_state_hook_scales_with_strength():
    gate = AffectiveGate(32, 8, mode="additive")
    aff = torch.randn(32)
    hidden = torch.zeros(1, 2, 8)
    s_lo, s_hi = 0.5, 2.0
    hook_lo = make_hidden_state_hook(gate, lambda: aff.clone(), strength=s_lo)
    hook_hi = make_hidden_state_hook(gate, lambda: aff.clone(), strength=s_hi)

    out_lo = hook_lo(None, None, hidden.clone())
    out_hi = hook_hi(None, None, hidden.clone())
    delta_lo = out_lo - hidden
    delta_hi = out_hi - hidden
    assert torch.allclose(delta_hi, (s_hi / s_lo) * delta_lo, atol=1e-5)


def test_affective_state_zero_and_get():
    state = AffectiveState(dim=32, device="cpu")
    z = state.zero()
    assert z.shape == (32,)
    assert torch.allclose(state.get(), z)


def test_clip_affect_norm_caps_magnitude():
    from src.affective.affect_norm import clip_affect_norm
    from src.config import GATE_MAX_AFFECT_NORM

    vec = np.ones(32, dtype=np.float32) * 10.0
    clipped = clip_affect_norm(vec)
    assert np.linalg.norm(clipped) <= GATE_MAX_AFFECT_NORM + 1e-5
