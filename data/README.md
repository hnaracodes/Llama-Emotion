# Data layout (Llama-Emotion / Spiking Affective Adapter)

This directory holds **small, eval-only** assets in git. The **training corpus is downloaded**; **trained weights** live on Modal volume `saa-models` (or a local mirror under `data/artifacts/`).

Full behavioral write-up: [`docs/results.md`](../docs/results.md). Interactive chat hardening plan: [`docs/chat_hardening_plan.md`](../docs/chat_hardening_plan.md).

---

## How it works (plain language)

Llama-Emotion does **not** fine-tune Llama's weights for emotion. Instead it runs a small **affective stack** beside a frozen W4-quantized Llama 3.2 1B, and injects a learned **additive bias** into hidden states during generation.

```
User text (chat transcript)
        │
        ▼
┌───────────────────────┐
│ AffectEncoder         │  MiniLM + head → 32-d vector per turn (VAD + macro emotions)
│ (EmpatheticDialogues) │
└───────────┬───────────┘
            │ trajectory over turns
            ▼
┌───────────────────────┐
│ LIF Amygdala (SNN)    │  temporal dynamics on delta spikes between vectors
└───────────┬───────────┘
            │ 32-d control signal
            ▼
┌───────────────────────┐
│ AffectiveGate         │  linear map: 32-d → Llama hidden bias (trained, ~few MB)
└───────────┬───────────┘
            │ forward hooks on Llama layers
            ▼
┌───────────────────────┐
│ Llama 3.2 1B (W4)     │  frozen; generates reply with affect-conditioned logits
└───────────────────────┘
```

**Gate v3 / v3.1 training objective:** optimize the gate so that, with hooks **on**, Llama's greedy generation is better at predicting the **human listener's next reply** in EmpatheticDialogues (listener CE), while staying **inert on neutral** inputs (neutral noop loss). Distress and neutral buckets are balanced so the gate cannot collapse into repeating empathy tokens (the v1/v2 failure mode).

**At chat time (`ChatEngine` / `chat.py`):** each user turn refreshes affect from the live transcript, registers hooks for that generation only, decodes **new tokens only** for the reply (not the full prompt), and removes hooks afterward. Strength `/strength`, manual override `/affect`, and periodic `/refresh` are exposed in the CLI.

**Runtime collapse guard (hardened 2026-07-04, `docs/chat_hardening_plan.md` Phase 1A):** every `generate_reply()` call runs the same `detect_empathy_collapse` / `collapse_score` check used in benchmarks on the freshly generated reply. If it fires and hooks were on, the engine retries once with hooks forced off; if it's still collapsed (or hooks were already off), it returns a fixed safe fallback string instead of raw looping text. `gate_health()` also verifies the loaded gate checkpoint is `trained` and matches `config.GATE_VERSION`, warning (local CLI/API) or fail-fast (Modal `EmotionalChatWorker.setup()`) otherwise.

---

## What we can claim (July 2026, Gate v3.1)

These statements are supported by Modal runs + 128 passing pytest tests after the v3.1 hardening and collapse-detector fixes:

| Claim | Evidence |
|-------|----------|
| **No empathy-token collapse** on training holdout | 500×2 retrain: `collapse_score: 0.0`, `any_collapse: false` from step 550 onward |
| **Gate is not inert** | Holdout records `text_changed: true` and non-zero `gate_output_norm`; Phase 4 `fraction_text_changed ≥ 0.8` |
| **Multi-turn chat benchmark is collapse-free** | Chat A/B (distress / neutral / hopeful): all `collapse_detected: false` with coherent replies |
| **Scenario holdout is collapse-free** | 29 scripted scenarios: hooks-on generation passes collapse check on **generated text only** |
| **Affect modulates output** | Chat A/B: distress/hopeful differ from neutral baseline (`text_changed`, empathy/sentiment deltas) |
| **Interactive chat has a runtime safety net (new 2026-07-04)** | `ChatEngine.generate_reply` runs the collapse detector on every turn (not just benchmarks), retries hooks-off, and falls back to a safe reply if still collapsed; gate checkpoint provenance is checked at load time. Unit-tested (no GPU) in `tests/test_chat_engine_collapse_guard.py` |

**Checkpoint tag:** `gate_version: v3.1_listener_ce_hardened` (see `train_gate.json` on Modal volume).

---

## What we cannot claim yet

| Limitation | Why |
|------------|-----|
| **Human-quality empathy** | Lexical/embedding metrics only; no human eval |
| **Long-session stability at 256 tokens/turn** | Runtime collapse guard + per-turn metrics are implemented and unit-tested (Phases 1-2, 4-5 of `docs/chat_hardening_plan.md`); the 10-turn Modal soak benchmark (`benchmark_phase_chat_soak.py`, Phase 3) is written but **not yet executed on GPU** — no empirical 256-token multi-turn collapse-rate number yet |
| **Teacher-forcing ↔ rollout gap** | Gate trains with gold listener tokens; inference uses model's own samples (standard exposure bias) |
| **Independent Phase 4 on every prompt** | Gate training holdout uses **disjoint** prompts from Phase 4 ablation set by design |

---

## Purpose

| Location | Role |
|----------|------|
| `data/` (committed) | Lexicon, holdout scenarios, pytest fixtures |
| `data/raw/` (gitignored) | EmpatheticDialogues CSVs — download before training |
| `data/artifacts/` (gitignored) | Local checkpoints and benchmark outputs |
| Modal `/models/` | Canonical GPU-trained encoder, SNN, gate weights |

## Committed in repo

| Path | Contents |
|------|----------|
| `data/lexicon/emotion_lexicon.json` | 32-d emotion prototypes (VAD + macro buckets) |
| `data/scenarios/*.json` | **29** holdout multi-turn scripts (eval only) |
| `tests/fixtures/empatheticdialogues/` | Tiny CSV slices for pytest |

## Holdout scenarios

Each file under `data/scenarios/` follows this schema:

```json
{
  "id": "tone_calm_to_panic",
  "category": "tone_shift",
  "emotion": "terrified",
  "tags": ["sudden_tone_shift"],
  "eval_question": "What should I do next?",
  "messages": [{"role": "user", "content": "..."}],
  "holdout": true,
  "source": "synthetic_template_v1"
}
```

| Category | Count | Use |
|----------|-------|-----|
| `distress_recovery` | 10 | Empathy / tone recovery |
| `conflict` | 7 | Tension / disagreement |
| `factual_neutral` | 6 | Low-affect planning (collapse probe) |
| `tone_shift` | 6 | Sudden valence changes |

**Regenerate:** `py -3 scripts/generate_scenarios.py`

**Holdout policy:** All `messages[].content` and `eval_question` strings are registered via `collect_holdout_texts()` in `src/affective/dataset.py`. They must **never** appear in encoder, SNN, or gate training splits.

## Downloaded (gitignored)

| Path | Source | Command |
|------|--------|---------|
| `data/raw/empatheticdialogues/` | [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (CC-BY-NC) | `py -3 scripts/download_empatheticdialogues.py` |

Expected files:

```
data/raw/empatheticdialogues/
  empchat_train.csv
  empchat_valid.csv
  empchat_test.csv
```

## Local artifacts (gitignored)

```
data/artifacts/
  affect/          # local encoder ckpt (optional; canonical on Modal)
  snn/
  gate/
  modal/           # pulled from volume (train_gate.json, holdout_eval.json)
  verification_report.json
  behavioral_verification_summary.json
  scenario_encoder_eval.json
  benchmarks/      # optional local mirror (phase_chat_ab.json, phase4_extended.json, …)
```

## Modal volume layout

Paths resolve via `src/runtime_paths.py`:

| Local mirror | Modal (`/models/`) |
|--------------|-------------------|
| `data/artifacts/affect/` | `/models/affect/` |
| `data/artifacts/snn/` | `/models/snn/` |
| `data/artifacts/gate/` | `/models/gate/` |
| `data/raw/empatheticdialogues/` | `/models/data/raw/empatheticdialogues/` |
| — | `/models/benchmarks/phase4_extended.json`, `phase_chat_ab.json`, etc. |

Baked into the Modal image at build time: `data/scenarios/` and `data/lexicon/` → `/opt/saa/data/` (see `src/common.py`).

---

## Quick start

**Modal (recommended — GPU + deps):**

```powershell
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
$env:PYTHONIOENCODING='utf-8'

py -3 -m modal run train_m1.py
py -3 -m modal run train_gate.py --max-samples 500 --epochs 2   # Gate v3.1
py -3 scripts/run_behavioral_verification.py --skip-train
py -3 -m modal run benchmark_phase_scenarios.py --max-new-tokens 64
py -3 -m modal run benchmark_phase_chat_ab.py --max-new-tokens 64
py -3 -m modal run benchmark_phase_chat_soak.py   # 10-turn, 256 tok/turn — not yet executed, see below
```

**Interactive emotional chat (CLI):**

```powershell
# Local CUDA GPU (needs ~4 GB VRAM for W4 1B + hooks)
py -3 chat.py --local

# Or Modal warm worker (no local GPU)
py -3 chat.py --modal

# In-session: /mood /colors /refresh /strength 1.0 /affect high|low|neutral /status /save /quit
```

`/status` prints gate checkpoint version/health and the last turn's collapse/affect diagnostics. If a turn's reply collapses, the CLI prints a `[collapse guard]` banner and either an auto-recovered (hooks-off retry) or safe-fallback reply — never raw looping text.

See [`docs/chat_hardening_plan.md`](../docs/chat_hardening_plan.md) for the full hardening plan and checklist (Phases 1, 2, 4, 5 implemented and unit-tested; Phase 3 GPU soak run still pending).

**Emotion Microscope API (local HTTP introspection):**

```powershell
py -3 -m pip install fastapi uvicorn
py -3 run_microscope.py
# POST http://localhost:8765/chat    {"message": "...", "session_id": "demo"}
# GET  http://localhost:8765/health/demo   -> gate provenance for that session
```

**Local (CI / fixture smoke tests):**

```powershell
py -3 -m pip install -r requirements.txt
py -3 scripts/download_empatheticdialogues.py
py -3 -m pytest tests/ -q
py -3 train_affect_encoder.py --local --fixture --epochs 2
```

`sentence-transformers/all-MiniLM-L6-v2` is fetched from Hugging Face on first run (`HF_HOME` cache). No need to commit base model weights.

---

## Interpreting benchmark JSON

When reading `data/artifacts/benchmarks/*.json` or Modal volume copies:

- **`collapse_detected`** must be computed on **`new_text`** (model output only), not the full decoded sequence. Prior to 2026-07-04, Chat A/B falsely flagged collapse because scripted transcript words ("feel", "here with you") were included — fixed in `generate_text` → `stats["new_text"]`.
- **`comparisons_vs_neutral`** in Chat A/B must compare **`new_text`** across scenarios (each scenario has a different scripted history). Comparing full sequences measured prompt differences, not model behavior — fixed in `benchmark_phase_chat_ab.py`.
- **`text_changed`** in Phase 4 / scenario hooks-on vs hooks-off is trustworthy (same prompt, different hook state).
- **`empathy_delta` / `sentiment_delta`** are lexical heuristics, not human judgments.

---

## Licenses

- **EmpatheticDialogues:** CC-BY-NC — training only; do not redistribute raw dumps.
- **Scenarios:** Synthetic holdout templates — safe to commit; not derived from ED utterances.
