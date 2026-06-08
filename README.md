# Spiking Affective Adapter

Hybrid brain-inspired system: **4-bit quantized Llama 3.2** (context stream) + **spiking affective amygdala** (SNN) modulating generation via compact **32-d float32** affective vectors derived from [TRIBEv2](https://huggingface.co/facebook/tribev2) fMRI predictions.

## Architecture

- **Context stream:** Llama-3.2-1B-Instruct with NF4 weights (bitsandbytes); KV memory via HF `QuantizedDynamicCache` (hook-compatible) + vLLM FP8 serving benchmarks (Phase 1b).
- **Affective stream:** TRIBEv2 → 32-d compression → delta-mod spikes → LIF SNN → neuromodulatory hooks on Llama layers.

## Prerequisites

1. [Modal](https://modal.com) account and CLI: `pip install modal && modal setup`
2. Hugging Face access to [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
3. Create Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=<your_huggingface_token>
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1a: 4-bit weight benchmark on Modal GPU
modal run benchmark_phase1a.py

# Phase 1b: KV cache (HF analytic + quantized cache)
modal run benchmark_phase1b.py

# Phase 1b: vLLM serving benchmark (auto vs fp8 KV, Python 3.12 image)
modal run benchmark_phase1b_vllm.py

# Phase 2–3: affective pipeline (synthetic or TRIBEv2)
modal run train_snn.py --stub

# Phase 4: hybrid modulated inference
modal run run_hybrid.py

# Phase 4 extended: strength sweep, logit KL, multi-prompt ablation
modal run benchmark_phase4_extended.py

# Phase chat A/B: transcript-conditioned affect comparison
modal run benchmark_phase_chat_ab.py

# Emotional CLI chat (local CUDA or Modal)
.venv\Scripts\python.exe chat.py --modal
.venv\Scripts\python.exe chat.py --local
```

## Quantization (important)

| Component | Phase | Method |
|-----------|-------|--------|
| **Weights** | 1a | bitsandbytes NF4 (`load_in_4bit=True`) |
| **KV cache (hooks)** | 1b | `QuantizedDynamicCache` INT8/INT4 in Hugging Face |
| **KV cache (serving)** | 1b | vLLM `kv_cache_dtype=auto` vs `fp8` |

Do not describe Phase 1a as "INT4 KV cache" — bnb quantizes weights only.

**Deep dive:** [docs/kv_cache_and_quantization.md](docs/kv_cache_and_quantization.md)

Optional vLLM deps: `pip install -r requirements-vllm.txt` (Modal `vllm_image` installs these automatically).

## Licenses

- **Llama 3.2:** Meta community license (gated on Hugging Face)
- **TRIBEv2:** CC-BY-NC-4.0 (non-commercial)

## Project layout

See `docs/benchmarks.md`.
