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

## Phase 4 extended — Strength sweep, logit KL, multi-prompt ablation

```bash
modal run benchmark_phase4_extended.py
modal run benchmark_phase4_extended.py --strengths 0,1,2,4
modal run benchmark_phase4_extended.py --skip-strength-sweep
```

| Metric | Description |
|--------|-------------|
| `strength_sweep[]` | Same prompt at `strength ∈ PHASE4_STRENGTH_SWEEP` with high affect vector |
| `strength_sweep[].logit_kl_*` | KL divergence at last prompt token (neutral vs high vector) |
| `prompt_ablation[]` | Per-prompt neutral vs high_affect vs hooks-off ablation |
| `neutral_vs_high_affect` | `text_changed`, lexical empathy/sentiment deltas, embedding cosine distance |
| `hooks_effect` | high vector with hooks on vs hooks off |
| `summary.fraction_text_changed` | Share of prompts where neutral ≠ high_affect text |

**Artifact:** `/artifacts/benchmarks/phase4_extended.json`

**Code:** `src/benchmark/affect_metrics.py`, `src/benchmark/hybrid_runner.py`

## Phase chat A/B — Transcript-conditioned affect

```bash
modal run benchmark_phase_chat_ab.py
```

| Metric | Description |
|--------|-------------|
| `scenarios.*.traits` | Affect traits from distress / neutral / hopeful transcripts |
| `scenarios.*.generation` | Reply to fixed user question under scenario affect |
| `comparisons_vs_neutral` | Lexical + embedding deltas vs neutral transcript baseline |
| `distress_vs_hopeful` | Mood-arc comparison (same opening, different closing turn) |

**Artifact:** `/artifacts/benchmarks/phase_chat_ab.json`

**Config:** `CHAT_AB_TRANSCRIPTS`, `CHAT_AB_USER_QUESTION` in `src/config.py`

## Phase 5 — STDP

Run `train_snn.py --stdp-steps N` and inspect `stdp_log[].delta_norm` in `train_snn.json`.

## Phase 6 — Summary table

| Phase | VRAM | Throughput | Affective ablation |
|-------|------|------------|-------------------|
| 1a | Yes | Yes | — |
| 1b | KV analytic + HF + vLLM | vLLM latency | — |
| 4 | Yes (W4) | Yes | neutral vs high_affect |
| 4 ext | Yes (W4) | Yes | strength sweep, logit KL, multi-prompt, hooks off |
| chat A/B | Yes (W4) | Yes | transcript scenarios vs neutral |
| 3/5 | — | SNN latency | firing rate stats |

## Local unit tests (no GPU)

```bash
pytest tests/ -q
```
