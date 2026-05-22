"""
Phase 1a: Modal benchmark for W4 (NF4) Llama 3.2 1B vs FP16 baseline.

Usage:
  modal run benchmark_phase1a.py
  modal run benchmark_phase1a.py --skip-fp16
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.common import app, gpu_kwargs, image, model_volume
from src.config import (
    ARTIFACTS_MOUNT,
    BENCHMARK_MAX_NEW_TOKENS,
    BENCHMARK_PROMPT,
    MODEL_ID,
)


@app.function(image=image, **gpu_kwargs())
def benchmark_w4_nf4(skip_fp16: bool = False) -> dict:
    import torch

    from src.llm.loader import generate_text, load_fp16_baseline, load_quantized_llama

    results: dict = {"model_id": MODEL_ID, "prompt": BENCHMARK_PROMPT, "runs": {}}

    model, tokenizer = load_quantized_llama()
    text, stats = generate_text(
        model,
        tokenizer,
        BENCHMARK_PROMPT,
        max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
        temperature=0.0,
    )
    results["runs"]["w4_nf4"] = {
        "quantization": "bnb_nf4_weights",
        "generated_preview": text[:500],
        **stats,
        "peak_vram_gb": stats["peak_vram_bytes"] / (1024**3),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not skip_fp16:
        model_fp, tok_fp = load_fp16_baseline()
        text_fp, stats_fp = generate_text(
            model_fp,
            tok_fp,
            BENCHMARK_PROMPT,
            max_new_tokens=BENCHMARK_MAX_NEW_TOKENS,
            temperature=0.0,
        )
        results["runs"]["fp16_baseline"] = {
            "quantization": "fp16_bfloat16_weights",
            "generated_preview": text_fp[:500],
            **stats_fp,
            "peak_vram_gb": stats_fp["peak_vram_bytes"] / (1024**3),
        }
        w4 = results["runs"]["w4_nf4"]["peak_vram_gb"]
        fp = results["runs"]["fp16_baseline"]["peak_vram_gb"]
        if fp > 0:
            results["vram_reduction_ratio"] = round(fp / w4, 3)

    out_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "phase1a_w4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_volume.commit()
    return results


@app.local_entrypoint()
def main(skip_fp16: bool = False):
    result = benchmark_w4_nf4.remote(skip_fp16=skip_fp16)
    print(json.dumps(result, indent=2))
