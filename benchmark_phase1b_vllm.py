"""
Phase 1b: vLLM KV-cache serving benchmarks (auto vs fp8).

Heavy Modal image with vLLM on Python 3.12. Merges into phase1b_kv.json on the volume.

Usage:
  modal run benchmark_phase1b_vllm.py
  modal run benchmark_phase1b.py && modal run benchmark_phase1b_vllm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import app, model_volume, vllm_gpu_kwargs, vllm_image
from src.config import ARTIFACTS_MOUNT, BENCHMARK_PROMPT, MODEL_ID


@app.function(image=vllm_image, **vllm_gpu_kwargs())
def benchmark_kv_vllm() -> dict:
    from src.llm.kv_benchmark import (
        load_phase1b_results,
        merge_vllm_into_phase1b_results,
        run_vllm_kv_benchmark,
        save_phase1b_results,
    )

    vllm_result = run_vllm_kv_benchmark(MODEL_ID, BENCHMARK_PROMPT)
    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase1b_kv.json"
    existing = load_phase1b_results(out_path)
    if existing:
        results = merge_vllm_into_phase1b_results(existing, vllm_result)
    else:
        results = {
            "model_id": MODEL_ID,
            "vllm_kv_benchmark": vllm_result,
            "notes": {
                "phase_1b_vllm": "vLLM kv_cache_dtype auto vs fp8 for production KV quant",
            },
        }
    save_phase1b_results(results, out_path)
    model_volume.commit()
    return results


@app.local_entrypoint()
def main():
    result = benchmark_kv_vllm.remote()
    print(json.dumps(result, indent=2))
