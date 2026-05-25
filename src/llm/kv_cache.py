"""
Quantized KV cache for Hugging Face Llama inference (Phase 1b).

Stores past Key/Value tensors in INT8 or INT4 instead of FP16/BF16, then
dequantizes on read so the model's attention math stays unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache, CacheLayerMixin, DynamicLayer
except ImportError:  # pragma: no cover
    Cache = object  # type: ignore
    CacheLayerMixin = object  # type: ignore
    DynamicLayer = object  # type: ignore


def _quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-token, per-head asymmetric quant along head_dim (common KV quant granularity).

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
    scale = scale.unsqueeze(-1).to(dtype=dtype)
    zero = zero.unsqueeze(-1).to(dtype=dtype)
    return q.to(dtype) * scale + zero


class QuantizedDynamicLayer(CacheLayerMixin):
    """Single-layer cache with INT8/INT4 (or FP16) K/V storage."""

    is_sliding = False

    def __init__(self, bits: int = 8, config=None) -> None:
        super().__init__()
        if bits not in (4, 8, 16):
            raise ValueError("bits must be 4, 8, or 16")
        self.bits = bits
        self.key_chunks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.value_chunks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._storage_bytes = 0

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _append_chunk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if self.bits == 16:
            k_store = key_states.detach()
            v_store = value_states.detach()
            self.key_chunks.append((k_store, torch.tensor(0), torch.tensor(0)))
            self.value_chunks.append((v_store, torch.tensor(0), torch.tensor(0)))
            self._storage_bytes += k_store.nbytes + v_store.nbytes
        else:
            k_pack = _quantize_asymmetric(key_states, self.bits)
            v_pack = _quantize_asymmetric(value_states, self.bits)
            self.key_chunks.append(k_pack)
            self.value_chunks.append(v_pack)
            for t in (*k_pack, *v_pack):
                self._storage_bytes += t.nbytes

    def _reconstruct(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.key_chunks:
            return self.keys, self.values

        if self.bits == 16:
            key_cat = torch.cat([c[0] for c in self.key_chunks], dim=-2)
            val_cat = torch.cat([c[0] for c in self.value_chunks], dim=-2)
            return key_cat, val_cat

        keys = [
            _dequantize_asymmetric(kq, ks, kz, dtype)
            for kq, ks, kz in self.key_chunks
        ]
        vals = [
            _dequantize_asymmetric(vq, vs, vz, dtype)
            for vq, vs, vz in self.value_chunks
        ]
        return torch.cat(keys, dim=-2), torch.cat(vals, dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._append_chunk(key_states, value_states)
        self.keys, self.values = self._reconstruct(self.dtype)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if not self.is_initialized or not self.key_chunks:
            return 0
        total = 0
        for chunk in self.key_chunks:
            total += chunk[0].shape[-2]
        return total

    def get_max_cache_shape(self) -> int:
        return -1

    def storage_bytes(self) -> int:
        return self._storage_bytes

    def reset_storage_counter(self) -> None:
        self._storage_bytes = 0


def _make_quantized_layer_class(bits: int) -> type:
    """Factory for Cache layer classes parameterized by quant bit-width."""

    class _QuantizedLayer(QuantizedDynamicLayer):
        def __init__(self, config=None) -> None:
            super().__init__(bits=bits, config=config)

    _QuantizedLayer.__name__ = f"QuantizedDynamicLayer{bits}"
    _QuantizedLayer.__qualname__ = _QuantizedLayer.__name__
    return _QuantizedLayer


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
        layer_class = DynamicLayer if bits == 16 else _make_quantized_layer_class(bits)
        super().__init__(layer_class_to_replicate=layer_class)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx or 0)

    def storage_bytes(self) -> int:
        """Bytes used by quantized (or fp) cache storage only."""
        total = 0
        for layer in self.layers:
            if isinstance(layer, QuantizedDynamicLayer):
                total += layer.storage_bytes()
            elif layer.is_initialized and layer.keys is not None and layer.values is not None:
                total += layer.keys.nbytes + layer.values.nbytes
        return total

    def reset_storage_counter(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_storage_counter"):
                layer.reset_storage_counter()


def fp16_cache_storage_bytes(cache) -> int:
    """Byte size of a standard DynamicCache's K+V tensors."""
    total = 0
    for layer in cache.layers:
        if layer.is_initialized and layer.keys is not None and layer.values is not None:
            total += layer.keys.nbytes + layer.values.nbytes
    return total


def cache_storage_bytes(cache) -> int:
    """
    Byte size of stored K/V for Hugging Face DynamicCache or QuantizedDynamicCache.

    Works with modern layer-based DynamicCache (no legacy key_cache attribute).
    """
    if hasattr(cache, "storage_bytes"):
        nbytes = cache.storage_bytes()
        if nbytes > 0:
            return nbytes
    if hasattr(cache, "layers"):
        return fp16_cache_storage_bytes(cache)
    # Legacy transformers layout
    if hasattr(cache, "key_cache") and cache.key_cache:
        total = 0
        for key_states, value_states in zip(cache.key_cache, cache.value_cache):
            if key_states is not None:
                total += key_states.nbytes
            if value_states is not None:
                total += value_states.nbytes
        return total
    return 0
