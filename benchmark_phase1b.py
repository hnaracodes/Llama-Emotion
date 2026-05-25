"""
Phase 1b: KV-cache benchmarks (analytic + HF QuantizedDynamicCache).

Light Modal image — no vLLM. For vLLM serving benchmarks use benchmark_phase1b_vllm.py.

Usage:
  modal run benchmark_phase1b.py
  modal run benchmark_phase1b.py --skip-8192
  modal run benchmark_phase1b.py --seq-lens 512,2048,8192
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import app, gpu_kwargs, image, model_volume
from src.config import ARTIFACTS_MOUNT, BENCHMARK_CONTEXT_LENGTHS, BENCHMARK_PROMPT, MODEL_ID


@app.function(image=image, **gpu_kwargs())
def benchmark_kv_hf(seq_lengths: list[int] | None = None) -> dict:
    from src.llm.kv_benchmark import run_phase1b_full, save_phase1b_results

    lengths = seq_lengths or list(BENCHMARK_CONTEXT_LENGTHS)
    results = run_phase1b_full(
        MODEL_ID,
        BENCHMARK_PROMPT,
        seq_lengths=lengths,
        run_vllm=False,
    )

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase1b_kv.json"
    save_phase1b_results(results, out_path)
    model_volume.commit()
    return results


def _parse_seq_lens(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


@app.local_entrypoint()
def main(skip_8192: bool = False, seq_lens: str = ""):
    if seq_lens:
        lengths = _parse_seq_lens(seq_lens)
    else:
        lengths = [
            sl
            for sl in BENCHMARK_CONTEXT_LENGTHS
            if not (skip_8192 and sl == 8192)
        ]
    result = benchmark_kv_hf.remote(lengths)
    print(json.dumps(result, indent=2))
