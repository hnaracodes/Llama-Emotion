"""Phase 1b: KV-cache analytics, HF quantized cache benchmarks, LMDeploy serving benchmarks."""

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
    filler = " word" * max(1, target_len // 2)
    text = (prompt + filler)[: target_len * 4]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=target_len,
    )
    return {k: v.to(device) for k, v in inputs.items()}


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

    from src.llm.kv_cache import QuantizedDynamicCache, fp16_cache_storage_bytes
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
        torch_dtype=dtype,
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

        if bits == 16 and hasattr(past, "key_cache"):
            storage = fp16_cache_storage_bytes(past)
        elif hasattr(past, "storage_bytes"):
            storage = past.storage_bytes()
        else:
            storage = 0

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


def run_lmdeploy_kv_benchmark(
    model_id: str,
    prompt: str,
    *,
    quant_policies: list[int] | None = None,
    session_len: int = 4096,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """
    LMDeploy Turbomind: quant_policy 0 = FP16 KV, 8 = INT8 KV, 4 = INT4 KV.

    See https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html
    """
    quant_policies = quant_policies or [0, 8, 4]
    results: dict[str, Any] = {
        "model_id": model_id,
        "engine": "lmdeploy.TurbomindEngineConfig",
        "session_len": session_len,
        "policies": {},
    }

    try:
        from lmdeploy import TurbomindEngineConfig, pipeline
    except ImportError as exc:
        results["error"] = f"lmdeploy not installed: {exc}"
        return results

    for policy in quant_policies:
        label = {0: "fp16_kv", 8: "int8_kv", 4: "int4_kv"}.get(policy, f"policy_{policy}")
        try:
            engine_config = TurbomindEngineConfig(
                quant_policy=policy,
                session_len=session_len,
                max_batch_size=1,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            pipe = pipeline(model_id, backend_config=engine_config)
            responses = pipe([prompt], max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - t0
            peak = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            text = responses[0].text if responses else ""
            results["policies"][label] = {
                "quant_policy": policy,
                "peak_vram_gb": round(peak, 4),
                "latency_sec": round(elapsed, 3),
                "response_preview": (text or "")[:300],
                "status": "ok",
            }
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            results["policies"][label] = {
                "quant_policy": policy,
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
    run_lmdeploy: bool = True,
) -> dict[str, Any]:
    """Run analytic table + HF quantized cache + optional LMDeploy benchmarks."""
    from src.config import BENCHMARK_CONTEXT_LENGTHS, MODEL_ID

    model_id = model_id or MODEL_ID
    seq_lengths = seq_lengths or [512, 2048]

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
            "phase_1b_lmdeploy": "Turbomind quant_policy 8/4 for production KV quant",
        },
    }

    if run_lmdeploy:
        payload["lmdeploy_kv_benchmark"] = run_lmdeploy_kv_benchmark(
            model_id,
            prompt,
            session_len=max(seq_lengths) * 2,
            max_new_tokens=32,
        )

    return payload


def save_phase1b_results(results: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
