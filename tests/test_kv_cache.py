"""Tests for KV quantize/dequant and storage accounting."""

import pytest
import torch

from src.llm.kv_cache import (
    QuantizedDynamicLayer,
    _dequantize_asymmetric,
    _quantize_asymmetric,
)

pytest.importorskip("transformers")


def test_quantize_roundtrip_int8():
    x = torch.randn(1, 4, 8, 32)
    q, scale, zero = _quantize_asymmetric(x, bits=8)
    x_hat = _dequantize_asymmetric(q, scale, zero, x.dtype)
    err = (x - x_hat).abs().mean()
    assert err < 0.15


def test_quantized_cache_storage_smaller_than_fp16():
    layer = QuantizedDynamicLayer(bits=8)
    k = torch.randn(1, 8, 16, 128, dtype=torch.float16)
    v = torch.randn(1, 8, 16, 128, dtype=torch.float16)
    layer.update(k, v)
    q_bytes = layer.storage_bytes()
    fp_bytes = k.nbytes + v.nbytes
    assert q_bytes < fp_bytes


def test_cache_grows_seq_length():
    layer = QuantizedDynamicLayer(bits=8)
    for _ in range(3):
        k = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        v = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        layer.update(k, v)
    assert layer.get_seq_length() == 12
