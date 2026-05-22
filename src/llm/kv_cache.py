"""
Quantized KV cache for Hugging Face Llama inference (Phase 1b).

Stores past Key/Value tensors in INT8 or INT4 instead of FP16/BF16, then
dequantizes on read so the model's attention math stays unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:  # pragma: no cover
    Cache = object  # type: ignore


def _quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-token, per-head asymmetric quant along head_dim (LMDeploy-style granularity).

    tensor: [batch, num_heads, seq_len, head_dim]
    Returns (q, scale, zero_point) where q is uint8 in [0, 2**bits - 1].
    """
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    x = tensor.float()
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = ((mx - mn) / qmax).clamp(min=1e-8)
    q = torch.round((x - mn) / scale).clamp(0, qmax).to(torch.uint8)
    # scale/zero: [batch, heads, seq] — broadcast over head_dim on dequant
    return q, scale.squeeze(-1), mn.squeeze(-1)


def _dequantize_asymmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct [batch, heads, seq, head_dim] from quantized storage."""
    out = q.to(dtype) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out


class QuantizedDynamicCache(Cache):
    """
    Drop-in DynamicCache that stores K/V in INT8 or INT4.

    Memory savings are in *stored* cache tensors; each forward still dequantizes
    the full layer cache for attention (naive path — production engines fuse quant attn).
    """

    def __init__(self, bits: int = 8) -> None:
        if bits not in (4, 8, 16):
            raise ValueError("bits must be 4, 8, or 16 (16 = no quant, baseline)")
        self.bits = bits
        self.key_chunks: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        self.value_chunks: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        self._seen_tokens = 0
        self._storage_bytes = 0

    def _append_chunk(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        while len(self.key_chunks) <= layer_idx:
            self.key_chunks.append([])
            self.value_chunks.append([])

        if self.bits == 16:
            k_store = key_states.detach()
            v_store = value_states.detach()
            self.key_chunks[layer_idx].append((k_store, torch.tensor(0), torch.tensor(0)))
            self.value_chunks[layer_idx].append((v_store, torch.tensor(0), torch.tensor(0)))
            self._storage_bytes += k_store.nbytes + v_store.nbytes
        else:
            k_pack = _quantize_asymmetric(key_states, self.bits)
            v_pack = _quantize_asymmetric(value_states, self.bits)
            self.key_chunks[layer_idx].append(k_pack)
            self.value_chunks[layer_idx].append(v_pack)
            for t in (*k_pack, *v_pack):
                self._storage_bytes += t.nbytes

    def _reconstruct_layer(
        self,
        layer_idx: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_chunks):
            raise IndexError(f"No cache for layer {layer_idx}")

        if self.bits == 16:
            key_cat = torch.cat([c[0] for c in self.key_chunks[layer_idx]], dim=-2)
            val_cat = torch.cat([c[0] for c in self.value_chunks[layer_idx]], dim=-2)
            return key_cat, val_cat

        keys = [
            _dequantize_asymmetric(kq, ks, kz, dtype)
            for kq, ks, kz in self.key_chunks[layer_idx]
        ]
        vals = [
            _dequantize_asymmetric(vq, vs, vz, dtype)
            for vq, vs, vz in self.value_chunks[layer_idx]
        ]
        return torch.cat(keys, dim=-2), torch.cat(vals, dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        self._append_chunk(layer_idx, key_states, value_states)
        dtype = key_states.dtype
        return self._reconstruct_layer(layer_idx, dtype)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self.key_chunks) or not self.key_chunks[layer_idx]:
            return 0
        total = 0
        for chunk in self.key_chunks[layer_idx]:
            total += chunk[0].shape[-2]
        return total

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def storage_bytes(self) -> int:
        """Bytes used by quantized (or fp) cache storage only."""
        return self._storage_bytes

    def reset_storage_counter(self) -> None:
        self._storage_bytes = 0


def fp16_cache_storage_bytes(cache) -> int:
    """Byte size of a standard DynamicCache's K+V tensors."""
    total = 0
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        total += k.nbytes + v.nbytes
    return total
