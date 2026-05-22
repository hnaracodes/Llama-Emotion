"""Load Llama 3.2 with bitsandbytes 4-bit (NF4) weight quantization."""

from __future__ import annotations

import os
from typing import Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def build_bnb_config(compute_dtype: torch.dtype | None = None) -> BitsAndBytesConfig:
    if compute_dtype is None:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_llama(
    model_id: str | None = None,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
    device_map: str | int | dict = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load Llama with W4 NF4 weights. KV cache remains full precision in this path."""
    from src.config import MODEL_CACHE_DIR, MODEL_ID

    model_id = model_id or MODEL_ID
    token = token or os.environ.get("HF_TOKEN")
    cache_dir = cache_dir or os.environ.get("HF_HOME", MODEL_CACHE_DIR)

    bnb_config = build_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        token=token,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()
    return model, tokenizer


def load_fp16_baseline(
    model_id: str | None = None,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
    device_map: str | int | dict = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """FP16 baseline for VRAM comparison (Phase 1a benchmarks)."""
    from src.config import MODEL_CACHE_DIR, MODEL_ID

    model_id = model_id or MODEL_ID
    token = token or os.environ.get("HF_TOKEN")
    cache_dir = cache_dir or os.environ.get("HF_HOME", MODEL_CACHE_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        token=token,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Tuple[str, dict[str, Any]]:
    """Run a single generation pass and return text + timing stats."""
    import time

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    stats = {
        "elapsed_sec": elapsed,
        "new_tokens": new_tokens,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }
    return text, stats


@torch.inference_mode()
def generate_with_kv_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    kv_bits: int = 16,
    max_new_tokens: int = 128,
) -> Tuple[str, dict[str, Any]]:
    """
    Generate with an explicit KV cache mode.

    kv_bits: 16 = standard DynamicCache (FP16/BF16 KV),
             8 or 4 = QuantizedDynamicCache storage (Phase 1b).
    """
    from transformers import DynamicCache

    from src.llm.kv_cache import QuantizedDynamicCache

    if kv_bits == 16:
        past = DynamicCache()
    elif kv_bits in (4, 8):
        past = QuantizedDynamicCache(bits=kv_bits)
    else:
        raise ValueError("kv_bits must be 16, 8, or 4")

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attn = inputs.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    import time

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        past_key_values=past,
        use_cache=True,
    )
    past = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        out = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

    elapsed = time.perf_counter() - t0
    all_ids = torch.cat([input_ids, *generated], dim=1)
    text = tokenizer.decode(all_ids[0], skip_special_tokens=True)

    from src.llm.kv_cache import fp16_cache_storage_bytes

    if kv_bits == 16 and hasattr(past, "key_cache"):
        storage = fp16_cache_storage_bytes(past)
    elif hasattr(past, "storage_bytes"):
        storage = past.storage_bytes()
    else:
        storage = 0

    stats = {
        "kv_bits": kv_bits,
        "kv_storage_mb": storage / (1024**2),
        "elapsed_sec": elapsed,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }
    return text, stats
