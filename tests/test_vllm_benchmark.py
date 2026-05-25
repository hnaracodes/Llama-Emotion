"""Tests for vLLM benchmark helper (mocked — no GPU)."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_run_vllm_kv_benchmark_import_error():
    from src.llm.kv_benchmark import run_vllm_kv_benchmark

    with patch.dict(sys.modules, {"vllm": None}):
        result = run_vllm_kv_benchmark("meta-llama/Llama-3.2-1B-Instruct", "hello")
    assert "error" in result


def test_merge_vllm_into_phase1b_results():
    from src.llm.kv_benchmark import merge_vllm_into_phase1b_results

    existing = {"model_id": "test", "notes": {"phase_1a": "w4"}}
    vllm = {"modes": {"auto_kv": {"status": "ok"}}}
    merged = merge_vllm_into_phase1b_results(existing, vllm)
    assert merged["vllm_kv_benchmark"] == vllm
    assert "phase_1b_vllm" in merged["notes"]


def test_run_vllm_kv_benchmark_success_mock():
    from src.llm.kv_benchmark import run_vllm_kv_benchmark

    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="generated text")]
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [mock_output]

    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    mock_vllm.SamplingParams = MagicMock()

    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        result = run_vllm_kv_benchmark(
            "meta-llama/Llama-3.2-1B-Instruct",
            "hello",
            kv_cache_dtypes=["auto"],
            max_new_tokens=8,
        )

    assert "modes" in result
    assert result["modes"]["auto_kv"]["status"] == "ok"
    assert "generated" in result["modes"]["auto_kv"]["response_preview"]
