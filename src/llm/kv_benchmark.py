"""Phase 1b: KV-cache analytics, HF quantized cache benchmarks, vLLM serving benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def estimate_kv_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    """Estimate KV cache size: 2 tensors (K,V) per layer."""
    per_layer = 2 * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_element
    return num_layers * per_layer


def kv_comparison_table(
    *,
    num_layers: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_lengths: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Compare FP16 vs INT8 vs INT4 KV at several context lengths."""
    from src.config import BENCHMARK_CONTEXT_LENGTHS

    seq_lengths = seq_lengths or BENCHMARK_CONTEXT_LENGTHS
    rows = []
    for seq_len in seq_lengths:
        fp16 = estimate_kv_bytes(
            num_layers, num_kv_heads, head_dim, seq_len, bytes_per_element=2
        )
        int8 = estimate_kv_bytes(
            num_layers, num_kv_heads, head_dim, seq_len, bytes_per_element=1
        )
        int4 = estimate_kv_bytes(
            num_layers, num_kv_heads, head_dim, seq_len, bytes_per_element=0.5
        )
        rows.append(
            {
                "seq_len": seq_len,
                "kv_fp16_gb": round(fp16 / (1024**3), 4),
                "kv_int8_gb": round(int8 / (1024**3), 4),
                "kv_int4_gb": round(int4 / (1024**3), 4),
                "int8_savings_vs_fp16": round(fp16 / int8, 2) if int8 else None,
                "int4_savings_vs_fp16": round(fp16 / int4, 2) if int4 else None,
            }
        )
    return rows


def _build_prefill_inputs(tokenizer, prompt: str, target_len: int, device: torch.device):
    """Build inputs with exactly ``target_len`` tokens (pad with a repeated filler token)."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] >= target_len:
        input_ids = input_ids[:, :target_len]
    else:
        need = target_len - input_ids.shape[1]
        filler_ids = tokenizer.encode(" the", add_special_tokens=False)
        filler_id = filler_ids[0] if filler_ids else (tokenizer.eos_token_id or 0)
        pad = torch.full((1, need), filler_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def run_hf_kv_cache_benchmark(
    model_id: str,
    prompt: str,
    seq_len: int,
    *,
    token: str | None = None,
    decode_tokens: int = 16,
    cache_modes: list[int] | None = None,
) -> dict[str, Any]:
    """
    Benchmark DynamicCache (16-bit) vs QuantizedDynamicCache (8/4-bit storage).

    Returns per-mode: logical KV storage bytes, peak VRAM, seq length, decode preview.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from src.llm.kv_cache import QuantizedDynamicCache, cache_storage_bytes
    from src.llm.loader import build_bnb_config

    cache_modes = cache_modes or [16, 8, 4]

    bnb = build_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        token=token,
        dtype=dtype,
    )
    model.eval()
    device = next(model.parameters()).device
    inputs = _build_prefill_inputs(tokenizer, prompt, seq_len, device)
    actual_len = inputs["input_ids"].shape[1]

    results: dict[str, Any] = {
        "model_id": model_id,
        "target_seq_len": seq_len,
        "actual_seq_len": actual_len,
        "decode_tokens": decode_tokens,
        "modes": {},
    }

    for bits in cache_modes:
        if bits == 16:
            past = DynamicCache()
            mode_name = "fp16_dynamic"
        else:
            past = QuantizedDynamicCache(bits=bits)
            mode_name = f"int{bits}_quantized_storage"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(decode_tokens):
                out = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_token = out.logits[:, -1:, :].argmax(dim=-1)

        elapsed = time.perf_counter() - t0
        peak = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )

        storage = cache_storage_bytes(past)

        preview = tokenizer.decode(next_token[0], skip_special_tokens=True)
        results["modes"][mode_name] = {
            "kv_bits": bits,
            "kv_storage_mb": round(storage / (1024**2), 3),
            "peak_vram_gb": round(peak, 4),
            "prefill_plus_decode_sec": round(elapsed, 3),
            "final_seq_len": past.get_seq_length() if hasattr(past, "get_seq_length") else None,
            "decode_tail_preview": preview[:80],
        }

        del past
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_vllm_kv_benchmark(
    model_id: str,
    prompt: str,
    *,
    kv_cache_dtypes: list[str] | None = None,
    max_new_tokens: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """
    vLLM serving benchmark: compare kv_cache_dtype auto (baseline) vs fp8.

    See https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
    """
    import os

    from src.config import (
        VLLM_GPU_MEMORY_UTILIZATION,
        VLLM_KV_CACHE_DTYPES,
        VLLM_MAX_MODEL_LEN,
        VLLM_MAX_NEW_TOKENS,
    )

    kv_cache_dtypes = kv_cache_dtypes or list(VLLM_KV_CACHE_DTYPES)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else VLLM_MAX_NEW_TOKENS
    max_model_len = max_model_len if max_model_len is not None else VLLM_MAX_MODEL_LEN
    gpu_memory_utilization = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else VLLM_GPU_MEMORY_UTILIZATION
    )
    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    results: dict[str, Any] = {
        "model_id": model_id,
        "engine": "vllm.LLM",
        "max_model_len": max_model_len,
        "modes": {},
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        results["error"] = f"vllm not installed: {exc}"
        return results

    dtype_labels = {"auto": "auto_kv", "fp8": "fp8_kv"}
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    for kv_dtype in kv_cache_dtypes:
        label = dtype_labels.get(kv_dtype, f"{kv_dtype}_kv")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            llm = LLM(
                model=model_id,
                kv_cache_dtype=kv_dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            outputs = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - t0
            peak = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            results["modes"][label] = {
                "kv_cache_dtype": kv_dtype,
                "peak_vram_gb": round(peak, 4),
                "latency_sec": round(elapsed, 3),
                "response_preview": (text or "")[:300],
                "status": "ok",
            }
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            results["modes"][label] = {
                "kv_cache_dtype": kv_dtype,
                "status": "error",
                "error": str(exc),
            }

    return results


def run_phase1b_full(
    model_id: str,
    prompt: str,
    *,
    token: str | None = None,
    seq_lengths: list[int] | None = None,
    run_vllm: bool = False,
) -> dict[str, Any]:
    """Run analytic table + HF quantized cache + optional vLLM benchmarks."""
    from src.config import BENCHMARK_CONTEXT_LENGTHS, MODEL_ID

    model_id = model_id or MODEL_ID
    seq_lengths = seq_lengths or list(BENCHMARK_CONTEXT_LENGTHS)

    # Llama 3.2 1B: 16 layers, 8 kv heads, 128 head dim
    analytic = kv_comparison_table(
        num_layers=16,
        num_kv_heads=8,
        head_dim=128,
        seq_lengths=BENCHMARK_CONTEXT_LENGTHS,
    )

    hf_runs = []
    for sl in seq_lengths:
        hf_runs.append(
            run_hf_kv_cache_benchmark(
                model_id,
                prompt,
                sl,
                token=token,
                decode_tokens=16,
            )
        )

    payload: dict[str, Any] = {
        "model_id": model_id,
        "analytic_kv_table": analytic,
        "hf_quantized_cache_benchmarks": hf_runs,
        "notes": {
            "phase_1a": "bitsandbytes NF4 quantizes model WEIGHTS only",
            "phase_1b_hf": "QuantizedDynamicCache stores K/V in INT8/INT4; dequant on read",
            "phase_1b_vllm": "vLLM kv_cache_dtype auto vs fp8 for production KV quant",
        },
    }

    if run_vllm:
        payload["vllm_kv_benchmark"] = run_vllm_kv_benchmark(model_id, prompt, token=token)

    return payload


def merge_vllm_into_phase1b_results(
    existing: dict[str, Any],
    vllm_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge vLLM benchmark into an existing phase1b_kv.json payload."""
    merged = dict(existing)
    merged["vllm_kv_benchmark"] = vllm_result
    notes = dict(merged.get("notes", {}))
    notes["phase_1b_vllm"] = "vLLM kv_cache_dtype auto vs fp8 for production KV quant"
    merged["notes"] = notes
    return merged


def load_phase1b_results(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_phase1b_results(results: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
