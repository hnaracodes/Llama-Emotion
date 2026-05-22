"""
Phase 1b: KV-cache quantization benchmarks (HF QuantizedDynamicCache + LMDeploy).

Usage:
  modal run benchmark_phase1b.py
  modal run benchmark_phase1b.py --no-lmdeploy
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import app, gpu_kwargs, image, model_volume
from src.config import ARTIFACTS_MOUNT, BENCHMARK_PROMPT, MODEL_ID


@app.function(image=image, **gpu_kwargs())
def benchmark_kv(run_lmdeploy: bool = True) -> dict:
    from src.llm.kv_benchmark import run_phase1b_full, save_phase1b_results

    results = run_phase1b_full(
        MODEL_ID,
        BENCHMARK_PROMPT,
        run_lmdeploy=run_lmdeploy,
    )

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase1b_kv.json"
    save_phase1b_results(results, out_path)
    model_volume.commit()
    return results


@app.local_entrypoint()
def main(run_lmdeploy: bool = True):
    result = benchmark_kv.remote(run_lmdeploy=run_lmdeploy)
    print(json.dumps(result, indent=2))
