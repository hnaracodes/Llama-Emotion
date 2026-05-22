"""Unit tests for KV analytic helpers."""

from src.llm.kv_benchmark import estimate_kv_bytes, kv_comparison_table


def test_estimate_kv_bytes_positive():
    b = estimate_kv_bytes(16, 8, 128, 2048, bytes_per_element=2)
    assert b > 0


def test_kv_table_rows():
    rows = kv_comparison_table(num_layers=16, num_kv_heads=8, head_dim=128, seq_lengths=[512, 1024])
    assert len(rows) == 2
    assert rows[0]["int8_savings_vs_fp16"] == 2.0
