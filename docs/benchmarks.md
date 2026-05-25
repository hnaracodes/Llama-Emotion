# Benchmarks — Spiking Affective Adapter

Reproduction commands and metric definitions for Phases 1–6.

## Prerequisites

```bash
# Use project .venv only — do not install deps globally
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux

modal secret create huggingface-secret HF_TOKEN=<token>
.venv\Scripts\pip install -r requirements-dev.txt   # includes pytest
.venv\Scripts\python.exe -m pytest tests/ -q
```

## Phase 1a — W4 NF4 weights

```bash
modal run benchmark_phase1a.py
modal run benchmark_phase1a.py --skip-fp16
```

| Metric | Description |
|--------|-------------|
| `peak_vram_gb` | `torch.cuda.max_memory_allocated()` during generation |
| `tokens_per_sec` | `new_tokens / elapsed_sec` |
| `vram_reduction_ratio` | FP16 peak / W4 peak (when FP16 baseline run) |

**Artifact:** `/artifacts/benchmarks/phase1a_w4.json` on Modal volume `saa-models`.

**Note:** bitsandbytes quantizes **weights** (NF4), not KV cache.

References:

- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [bitsandbytes quickstart](https://huggingface.co/docs/bitsandbytes/main/quickstart)

## Phase 1b — KV cache

### HF path (hook-compatible, light Modal image)

```bash
modal run benchmark_phase1b.py
modal run benchmark_phase1b.py --skip-8192
modal run benchmark_phase1b.py --seq-lens 512,2048,8192
```

Default HF live runs use `BENCHMARK_CONTEXT_LENGTHS` (**512, 2048, 8192**). The 8192 run stress-tests long-context KV; skip it for faster iteration.

| Metric | Description |
|--------|-------------|
| `analytic_kv_table` | Theoretical FP16 / INT8 / INT4 KV size vs `seq_len` |
| `hf_quantized_cache_benchmarks` | `QuantizedDynamicCache` INT8/INT4 vs `DynamicCache` FP16 |
| `modes.*.kv_storage_mb` | Stored K/V bytes (`cache_storage_bytes`; FP16 uses layer tensors) |
| `actual_seq_len` | Should match `target_seq_len` after prefill padding fix |

### vLLM serving path (Python 3.12, separate image)

```bash
modal run benchmark_phase1b_vllm.py
```

| Metric | Description |
|--------|-------------|
| `vllm_kv_benchmark.modes.auto_kv` | Baseline `kv_cache_dtype=auto` |
| `vllm_kv_benchmark.modes.fp8_kv` | FP8 KV cache (`kv_cache_dtype=fp8`) |

Run both and merge into one artifact:

```bash
modal run benchmark_phase1b.py && modal run benchmark_phase1b_vllm.py
```

**Code:** `src/llm/kv_cache.py`, `src/llm/kv_benchmark.py`

**Artifact:** `/artifacts/benchmarks/phase1b_kv.json`

**Guide:** [kv_cache_and_quantization.md](kv_cache_and_quantization.md)

### JSON schema note (breaking change from LMDeploy era)

| Old key | New key |
|---------|---------|
| `lmdeploy_kv_benchmark` | `vllm_kv_benchmark` |
| `policies.int8_kv` / `int4_kv` | `modes.fp8_kv` |
| `quant_policy` | `kv_cache_dtype` |

References:

- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [KVQuant](https://github.com/SqueezeAILab/KVQuant)

## Phase 2–3 — Affective + SNN

```bash
modal run train_snn.py --stub
modal run train_snn.py --stdp-steps 50
```

| Metric | Description |
|--------|-------------|
| `spike_sparsity` | Fraction of timesteps with any spike |
| `mean_firing_rate` | SNN output layer activity |
| `affective_vector` | 32-d float32 state |

**Artifacts:** `/artifacts/snn/*.npy`, `amygdala.pt`, `train_snn.json`

## Phase 4 — Hybrid modulation

```bash
modal run run_hybrid.py --strength 1.0
```

| Metric | Description |
|--------|-------------|
| `conditions.neutral` | Generation with zero affective vector |
| `conditions.high_affect` | Generation with amplified TRIBEv2/SNN vector |
| `generated_preview` | Compare text shifts between conditions |

**Artifact:** `/artifacts/benchmarks/phase4_hybrid.json`

## Phase 5 — STDP

Run `train_snn.py --stdp-steps N` and inspect `stdp_log[].delta_norm` in `train_snn.json`.

## Phase 6 — Summary table

| Phase | VRAM | Throughput | Affective ablation |
|-------|------|------------|-------------------|
| 1a | Yes | Yes | — |
| 1b | KV analytic + HF + vLLM | vLLM latency | — |
| 4 | Yes (W4) | Yes | neutral vs high_affect |
| 3/5 | — | SNN latency | firing rate stats |

## Local unit tests (no GPU)

```bash
pytest tests/ -q
```
