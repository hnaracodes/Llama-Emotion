"""Tests for KV quantize/dequant and storage accounting."""

import torch

from src.llm.kv_cache import (
    QuantizedDynamicCache,
    _dequantize_asymmetric,
    _quantize_asymmetric,
)


def test_quantize_roundtrip_int8():
    x = torch.randn(1, 4, 8, 32)
    q, scale, zero = _quantize_asymmetric(x, bits=8)
    x_hat = _dequantize_asymmetric(q, scale, zero, x.dtype)
    err = (x - x_hat).abs().mean()
    assert err < 0.15


def test_quantized_cache_storage_smaller_than_fp16():
    cache_q = QuantizedDynamicCache(bits=8)
    k = torch.randn(1, 8, 16, 128, dtype=torch.float16)
    v = torch.randn(1, 8, 16, 128, dtype=torch.float16)
    cache_q.update(k, v, layer_idx=0)
    q_bytes = cache_q.storage_bytes()
    fp_bytes = k.nbytes + v.nbytes
    assert q_bytes < fp_bytes


def test_cache_grows_seq_length():
    cache = QuantizedDynamicCache(bits=8)
    for _ in range(3):
        k = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        v = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 12
