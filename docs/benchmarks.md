# Benchmarks — Spiking Affective Adapter

Reproduction commands and metric definitions for Phases 1–6.

## Prerequisites

```bash
modal secret create saa-hf-secret HF_TOKEN=<token>
pip install -r requirements.txt
pytest tests/ -q
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

## Phase 1b — KV cache (implemented)

```bash
modal run benchmark_phase1b.py
modal run benchmark_phase1b.py --no-lmdeploy
```

| Metric | Description |
|--------|-------------|
| `analytic_kv_table` | Estimated FP16 / INT8 / INT4 KV size vs `seq_len` |
| `hf_quantized_cache_benchmarks` | `QuantizedDynamicCache` INT8/INT4 vs `DynamicCache` FP16: `kv_storage_mb`, `peak_vram_gb` |
| `lmdeploy_kv_benchmark` | `quant_policy` 0 / 8 / 4 via Turbomind (if model supported) |

**Code:** `src/llm/kv_cache.py`, `src/llm/kv_benchmark.py`

**Artifact:** `/artifacts/benchmarks/phase1b_kv.json`

**Guide:** [kv_cache_and_quantization.md](kv_cache_and_quantization.md)

References:

- [LMDeploy KV quant](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
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
| 1b | KV analytic + growth | — | — |
| 4 | Yes (W4) | Yes | neutral vs high_affect |
| 3/5 | — | SNN latency | firing rate stats |

## Local unit tests (no GPU)

```bash
pytest tests/ -q
```
