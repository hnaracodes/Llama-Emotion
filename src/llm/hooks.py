"""Neuromodulatory hooks: affective 32-d vector → Llama hidden-state gate."""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn


class AffectiveGate(nn.Module):
    """Maps affective vector (D) → hidden_size bias/scale for residual stream."""

    def __init__(self, affect_dim: int, hidden_size: int, mode: str = "additive"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(affect_dim, hidden_size, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, affective: torch.Tensor) -> torch.Tensor:
        """
        affective: (D,) or (B, D)
        Returns modulation tensor (hidden_size,) or (B, hidden_size)
        """
        if affective.dim() == 1:
            affective = affective.unsqueeze(0)
        mod = self.proj(affective.float())
        if self.mode == "scale":
            return 1.0 + 0.1 * torch.tanh(mod)
        return mod  # additive


def make_hidden_state_hook(
    gate: AffectiveGate,
    get_affective: Callable[[], torch.Tensor],
    strength: float = 1.0,
) -> Callable:
    """Forward hook on LlamaDecoderLayer: add affective bias to hidden states."""

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()
        aff = get_affective()
        if aff is None:
            return output
        device = hidden.device
        mod = gate(aff.to(device))
        mod = mod.to(dtype=hidden.dtype)
        if mod.dim() == 1:
            mod = mod.unsqueeze(0).unsqueeze(0)
        elif mod.dim() == 2:
            mod = mod.unsqueeze(1)
        if gate.mode == "scale":
            hidden = hidden * mod
        else:
            hidden = hidden + strength * mod
        if rest:
            return (hidden,) + rest
        return hidden

    return hook


def register_affective_hooks(
    model,
    gate: AffectiveGate,
    get_affective: Callable[[], torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    strength: float = 1.0,
) -> List:
    """
    Register hooks on decoder layers (default: last 2 layers).
    Returns hook handles — call .remove() when done.
    """
    layers = model.model.layers
    if layer_indices is None:
        layer_indices = [len(layers) - 2, len(layers) - 1]
    handles = []
    hook_fn = make_hidden_state_hook(gate, get_affective, strength=strength)
    for idx in layer_indices:
        if 0 <= idx < len(layers):
            h = layers[idx].register_forward_hook(hook_fn)
            handles.append(h)
    return handles


def patch_attention_pre_softmax(layer, bias_fn: Callable[[], torch.Tensor]):
    """
    Advanced (Phase 4b): wrap LlamaAttention.forward to add bias before softmax.
    Disable FlashAttention on instrumented layers. bias_fn returns (num_heads,) or scalar.
    """
    import math

    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import repeat_kv

    attn = layer.self_attn
    original_forward = attn.forward

    def forward_with_bias(hidden_states, *args, **kwargs):
        # Reuse standard path up to attn_weights (simplified; no cache for brevity)
        bsz, q_len, _ = hidden_states.size()
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, attn.num_heads, attn.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, attn.num_key_value_heads, attn.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, attn.num_key_value_heads, attn.head_dim
        ).transpose(1, 2)
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(attn.head_dim)
        bias = bias_fn()
        if bias is not None:
            attn_weights = attn_weights + bias.to(attn_weights.device).view(
                1, -1, 1, 1
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            bsz, q_len, -1
        )
        attn_output = attn.o_proj(attn_output)
        return attn_output, None, None

    attn.forward = forward_with_bias
    return original_forward


class AffectiveState:
    """Mutable container for current affective vector during generation."""

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self._vec: Optional[torch.Tensor] = None

    def set(self, vec: torch.Tensor | None) -> None:
        self._vec = vec

    def get(self) -> Optional[torch.Tensor]:
        return self._vec

    def zero(self) -> torch.Tensor:
        z = torch.zeros(self.dim, device=self.device)
        self._vec = z
        return z
