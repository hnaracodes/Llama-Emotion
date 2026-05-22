# KV cache and quantization (low-level guide)

This document explains what Phase 1a and Phase 1b do in the Spiking Affective Adapter project, how they differ, and why both matter for long conversations.

---

## Two different memories in an LLM

When Llama generates text, GPU memory holds two big things:

| What | Stored where | Grows with… | Phase in this repo |
|------|----------------|-------------|-------------------|
| **Model weights** | Parameters (billions) | Model size (1B, 3B, …) | **1a** — 4-bit NF4 (bitsandbytes) |
| **KV cache** | Activations per layer | **Context length** (prompt + generated tokens) | **1b** — INT8 / INT4 KV |

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

## What is INT4 / INT8 KV quantization?

### Full precision KV (FP16 / BF16)

Each number in K and V is stored as 16 bits. High fidelity; largest memory.

### Quantization idea

Store K/V with fewer bits (8 or 4) using **scale** and **zero-point** (asymmetric quant):

```text
quantized = round((value - min) / scale)   clamped to 0 … 2^bits-1
reconstructed ≈ quantized * scale + min
```

This project uses **per-token, per-head** ranges along `head_dim` (same granularity LMDeploy documents).

### INT8 KV

- 1 byte per stored value → **~2× smaller** than FP16 KV.
- Usually small quality loss for Llama-class models.

### INT4 KV

- 4 bits per value → **~4× smaller** than FP16 KV.
- Strong memory win; some models degrade badly at INT4 KV (test on your model).

### Why it works (usually)

Attention only needs K/V to be **close enough** that softmax rankings stay similar. Small errors in stored K/V slightly perturb scores but rarely change the argmax token every step—especially at INT8.

### Why it’s useful *here*

1. **Long affective sessions** — Stream 2 + long prompts → you want headroom for context without OOM.
2. **Modal GPU cost** — Smaller KV → longer context on L4/A10.
3. **Complements 1a** — W4 weights + INT8 KV is a practical stack: cheap weights *and* cheap memory for history.

---

## Phase 1a vs Phase 1b in code

### Phase 1a — `src/llm/loader.py` + `benchmark_phase1a.py`

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
```

- **NF4** packs each weight into 4 bits with a learned-ish normal distribution codebook.
- Matmuls often run in FP16/BF16 **compute** dtype; weights stay compressed.
- **KV tensors are still created at full precision** in the default Hugging Face path.

### Phase 1b — three implementations in this repo

| Path | File | What it does |
|------|------|----------------|
| **Analytic** | `kv_benchmark.kv_comparison_table` | Math estimate of KV size at 512/2k/8k tokens |
| **HF quantized cache** | `src/llm/kv_cache.py` → `QuantizedDynamicCache` | Stores K/V as INT8/INT4; dequantizes when layer reads cache |
| **LMDeploy** | `kv_benchmark.run_lmdeploy_kv_benchmark` | `TurbomindEngineConfig(quant_policy=8 or 4)` — production-style serving |

Run on Modal:

```bash
modal run benchmark_phase1b.py
modal run benchmark_phase1b.py --no-lmdeploy   # skip LMDeploy if model unsupported
```

Results: `/artifacts/benchmarks/phase1b_kv.json` on volume `saa-models`.

---

## How `QuantizedDynamicCache` works (HF path)

1. On each `update()`, new K/V blocks are **quantized** to uint8 (+ scale/zero tensors).
2. Chunks are appended per layer (same API as `DynamicCache`).
3. On read, chunks are **dequantized** and concatenated so Llama’s attention code is unchanged.
4. `storage_bytes()` reports **compressed** cache size (compare to FP16 `DynamicCache`).

**Caveat:** This naive path still dequantizes full K/V during forward (temporary FP tensors). **Peak VRAM** may not drop as much as LMDeploy’s fused kernels—but **stored** KV size and the INT8/INT4 *concept* are measured correctly for research.

**Phase 4 hooks** stay on Hugging Face + this cache until you standardize on LMDeploy for serving.

---

## LMDeploy `quant_policy`

| `quant_policy` | KV storage |
|----------------|------------|
| `0` | FP16 (default) |
| `8` | INT8 KV |
| `4` | INT4 KV |

Reference: [LMDeploy KV quant docs](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)

---

## End-to-end data flow (Phase 1b benchmark)

```text
Prompt text
    → tokenize (seq_len tokens)
    → For each cache mode in {FP16, INT8, INT4}:
          past_key_values = DynamicCache() or QuantizedDynamicCache(bits)
          forward(prompt)     # fills cache
          forward N decode steps
          record: kv_storage_mb, peak_vram_gb, latency, text tail
    → (optional) LMDeploy pipeline with quant_policy 0/8/4
    → JSON artifact
```

---

## Reading benchmark JSON

Example fields:

- `analytic_kv_table` — theoretical GB at various seq lengths.
- `hf_quantized_cache_benchmarks[].modes.int8_quantized_storage.kv_storage_mb` — stored cache size.
- `hf_quantized_cache_benchmarks[].modes.fp16_dynamic.kv_storage_mb` — baseline.
- `lmdeploy_kv_benchmark.policies.int8_kv` — serving path with real `quant_policy`.

Compare **ratio** `fp16 / int8` storage (~2×) and `fp16 / int4` (~4×) to validate Phase 1b.

---

## Further reading

- [Transformers bitsandbytes (weights)](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [bitsandbytes quickstart](https://huggingface.co/docs/bitsandbytes/main/quickstart)
- [LMDeploy INT4/INT8 KV](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- [KVQuant research](https://github.com/SqueezeAILab/KVQuant)
