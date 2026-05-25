# KV cache and quantization (low-level guide)

This document explains what Phase 1a and Phase 1b do in the Spiking Affective Adapter project, how they differ, and why both matter for long conversations.

---

## Two different memories in an LLM

When Llama generates text, GPU memory holds two big things:

| What | Stored where | Grows with… | Phase in this repo |
|------|----------------|-------------|-------------------|
| **Model weights** | Parameters (billions) | Model size (1B, 3B, …) | **1a** — 4-bit NF4 (bitsandbytes) |
| **KV cache** | Activations per layer | **Context length** (prompt + generated tokens) | **1b** — HF INT8/INT4 + vLLM FP8 |

Confusing them is common: **Phase 1a does not shrink the KV cache.** It shrinks **weights**. Phase 1b targets **KV** specifically.

---

## What is the KV cache?

### Attention recap

Each transformer layer computes **attention**:

1. From hidden states, build **Query (Q)**, **Key (K)**, **Value (V)** vectors per token and per head.
2. Compare Q of the *current* token to K of *all past* tokens → attention scores.
3. Weighted sum of V → output.

For autoregressive generation you append one token at a time. Recomputing K and V for every old token every step would be O(n²) waste.

### The cache trick

After processing token *i*, save that layer’s **K** and **V** tensors. When token *i+1* arrives, only compute K/V for the new token and **append** to the cache. Attention reads **cached K/V** for old positions.

So the KV cache is literally:

```text
Per layer ℓ:
  Key_cache[ℓ]   shape ≈ [batch, num_heads, seq_len, head_dim]
  Value_cache[ℓ] shape ≈ [batch, num_heads, seq_len, head_dim]
```

**seq_len** = prompt length + tokens generated so far. Long chat → **seq_len grows** → KV dominates VRAM.

### Rough size formula (Llama 3.2 1B)

- 16 layers, 8 KV heads, head_dim 128, FP16 (2 bytes/element)
- KV bytes ≈ `2 × 16 × seq_len × 8 × 128 × 2` = `seq_len × 65,536` bytes (~64 KB per token)

At **32k tokens**, KV alone is on the order of **~2 GB** before weights and activations.

---

## What is KV quantization?

### Full precision KV (FP16 / BF16)

Each number in K and V is stored as 16 bits. High fidelity; largest memory.

### Integer quant (HF `QuantizedDynamicCache`)

Store K/V with 8 or 4 bits using **scale** and **zero-point** (asymmetric quant):

```text
quantized = round((value - min) / scale)   clamped to 0 … 2^bits-1
reconstructed ≈ quantized * scale + min
```

This project uses **per-token, per-head** ranges along `head_dim`.

| Mode | Storage | vs FP16 KV |
|------|---------|------------|
| INT8 | 1 byte/value | ~2× smaller |
| INT4 | 0.5 byte/value | ~4× smaller |

The **analytic table** in benchmarks uses these theoretical sizes.

### FP8 quant (vLLM serving)

Production serving stacks (vLLM) typically use **FP8 KV cache** (`kv_cache_dtype=fp8`) with fused attention kernels—not classic INT4. FP8 gives ~2× KV savings vs FP16 with strong accuracy on many Llama models.

Reference: [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)

### Why it works (usually)

Attention only needs K/V to be **close enough** that softmax rankings stay similar. Small errors in stored K/V slightly perturb scores but rarely change the argmax token every step.

### Why it’s useful *here*

1. **Long affective sessions** — Stream 2 + long prompts → headroom for context without OOM.
2. **Modal GPU cost** — Smaller KV → longer context on L4/A10.
3. **Complements 1a** — W4 weights + compressed KV is a practical stack.

---

## Phase 1a vs Phase 1b in code

### Phase 1a — `src/llm/loader.py` + `benchmark_phase1a.py`

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
```

- **NF4** packs weights into 4 bits.
- **KV tensors remain full precision** in the default Hugging Face path.

### Phase 1b — dual path in this repo

| Path | File | What it does |
|------|------|----------------|
| **Analytic** | `kv_benchmark.kv_comparison_table` | Theoretical FP16 / INT8 / INT4 sizes |
| **HF quantized cache** | `src/llm/kv_cache.py` → `QuantizedDynamicCache` | INT8/INT4 storage; **Phase 4 hooks compatible** |
| **vLLM serving** | `kv_benchmark.run_vllm_kv_benchmark` | `kv_cache_dtype=auto` vs `fp8` on Python 3.12 Modal image |

Run on Modal:

```bash
modal run benchmark_phase1b.py           # analytic + HF at 512, 2048, 8192
modal run benchmark_phase1b.py --skip-8192
modal run benchmark_phase1b_vllm.py      # vLLM auto vs fp8 (heavy image)
```

Results merge into `/artifacts/benchmarks/phase1b_kv.json` on volume `saa-models`.

---

## How `QuantizedDynamicCache` works (HF path)

1. On each `update()`, new K/V blocks are **quantized** to uint8 (+ scale/zero tensors).
2. Chunks are appended per layer (same API as `DynamicCache`).
3. On read, chunks are **dequantized** and concatenated so Llama’s attention code is unchanged.
4. `storage_bytes()` reports **compressed** cache size.

**Caveat:** This path dequantizes during forward. **Peak VRAM** may not drop as much as vLLM’s fused FP8 kernels—but stored KV size is measured correctly, and **hooks keep working**.

**Phase 4 hooks** stay on Hugging Face + this cache. vLLM is for serving benchmarks only until modulation is validated on a serving stack.

---

## Production serving (vLLM)

| `kv_cache_dtype` | Meaning |
|------------------|---------|
| `auto` | Model default (typically BF16/FP16 KV) |
| `fp8` | FP8 KV cache + FP8 attention where supported |

Configured in [`src/config.py`](../src/config.py) as `VLLM_KV_CACHE_DTYPES = ["auto", "fp8"]`.

Modal uses a separate **`vllm_image`** (Python 3.12) — see [`src/common.py`](../src/common.py).

---

## End-to-end data flow (Phase 1b benchmark)

```text
Prompt text
    → HF path: tokenize → DynamicCache vs QuantizedDynamicCache → metrics
    → vLLM path: LLM(kv_cache_dtype=auto|fp8) → generate → metrics
    → merge into phase1b_kv.json
```

---

## Reading benchmark JSON

- `analytic_kv_table` — theoretical GB at various seq lengths (INT8/INT4 math).
- `hf_quantized_cache_benchmarks[].modes.*.kv_storage_mb` — HF stored cache size.
- `vllm_kv_benchmark.modes.auto_kv` / `fp8_kv` — serving path with `peak_vram_gb`, `latency_sec`, `response_preview`.

**Breaking change:** `lmdeploy_kv_benchmark` → `vllm_kv_benchmark` (see [benchmarks.md](benchmarks.md)).

---

## Further reading

- [Transformers bitsandbytes (weights)](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [KVQuant research](https://github.com/SqueezeAILab/KVQuant)
