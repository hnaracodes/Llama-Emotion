# Spiking Affective Adapter

Hybrid brain-inspired system: **4-bit quantized Llama 3.2** (context stream) + **spiking affective amygdala** (SNN) modulating generation via compact **32-d float32** affective vectors derived from [TRIBEv2](https://huggingface.co/facebook/tribev2) fMRI predictions.

## Architecture

- **Context stream:** Llama-3.2-1B-Instruct with NF4 weights (bitsandbytes); optional KV INT8/INT4 via LMDeploy (Phase 1b).
- **Affective stream:** TRIBEv2 → 32-d compression → delta-mod spikes → LIF SNN → neuromodulatory hooks on Llama layers.

## Prerequisites

1. [Modal](https://modal.com) account and CLI: `pip install modal && modal setup`
2. Hugging Face access to [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
3. Create Modal secret:

```bash
modal secret create saa-hf-secret HF_TOKEN=<your_huggingface_token>
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1a: 4-bit weight benchmark on Modal GPU
modal run benchmark_phase1a.py

# Phase 1b: KV cache memory comparison
modal run benchmark_phase1b.py

# Phase 2–3: affective pipeline (synthetic or TRIBEv2)
modal run train_snn.py --stub

# Phase 4: hybrid modulated inference
modal run run_hybrid.py
```

## Quantization (important)

| Component | Phase | Method |
|-----------|-------|--------|
| **Weights** | 1a | bitsandbytes NF4 (`load_in_4bit=True`) |
| **KV cache** | 1b | `QuantizedDynamicCache` (INT8/INT4) in Hugging Face + LMDeploy `quant_policy` benchmarks |

Do not describe Phase 1a as "INT4 KV cache" — bnb quantizes weights only.

**Deep dive:** [docs/kv_cache_and_quantization.md](docs/kv_cache_and_quantization.md) — what KV cache is, why INT4/INT8 help long context, and how Phase 1b is implemented.

## Licenses

- **Llama 3.2:** Meta community license (gated on Hugging Face)
- **TRIBEv2:** CC-BY-NC-4.0 (non-commercial)

## Project layout

See `implementation_plan.md` and `docs/benchmarks.md`.
