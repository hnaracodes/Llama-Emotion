"""Unit tests for KV analytic helpers and prefill builder."""

import torch

from src.llm.kv_benchmark import _build_prefill_inputs, estimate_kv_bytes, kv_comparison_table
from src.llm.kv_cache import cache_storage_bytes, fp16_cache_storage_bytes


class _StubTokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **kwargs):
        del text, add_special_tokens, kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [99]


class _LayerStub:
    def __init__(self) -> None:
        self.is_initialized = True
        self.keys = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        self.values = torch.randn(1, 8, 4, 128, dtype=torch.float16)


class _CacheStub:
    def __init__(self, layers=None, storage: int | None = None) -> None:
        self.layers = layers or [_LayerStub()]
        self._storage = storage

    def storage_bytes(self) -> int:
        return self._storage or 0


def test_estimate_kv_bytes_positive():
    b = estimate_kv_bytes(16, 8, 128, 2048, bytes_per_element=2)
    assert b > 0


def test_kv_table_rows():
    rows = kv_comparison_table(num_layers=16, num_kv_heads=8, head_dim=128, seq_lengths=[512, 1024])
    assert len(rows) == 2
    assert rows[0]["int8_savings_vs_fp16"] == 2.0


def test_build_prefill_inputs_exact_length():
    tok = _StubTokenizer()
    inputs = _build_prefill_inputs(tok, "prompt", target_len=64, device=torch.device("cpu"))
    assert inputs["input_ids"].shape == (1, 64)
    assert inputs["attention_mask"].shape == (1, 64)


def test_cache_storage_bytes_prefers_quantized_storage():
    cache = _CacheStub(storage=2048)
    assert cache_storage_bytes(cache) == 2048


def test_cache_storage_bytes_fp16_from_layers():
    cache = _CacheStub()
    fp_bytes = cache_storage_bytes(cache)
    assert fp_bytes == fp16_cache_storage_bytes(cache)
    assert fp_bytes > 0
