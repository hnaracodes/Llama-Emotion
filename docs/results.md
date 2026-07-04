# M1 Training Results — EmpatheticDialogues v1 Supervision

> **Session:** June 2026 · **Milestone:** M1 ("Real, not random")  
> **Platform:** Modal GPU (L4), volume `saa-models`  
> **Supervision:** `empatheticdialogues_v1` — labeled dialogue, **not** TRIBEv2/fMRI  
> **Plan reference:** [teaching_ai_emotion_plan.md](./teaching_ai_emotion_plan.md)

This document records metrics, failures, fixes, and honest interpretation from the first
end-to-end training of the v1 affective stack: **encoder → SNN amygdala → affective gate**.

---

## Executive summary

| Component | Status | Primary metric | Honest read |
|-----------|--------|--------------|-------------|
| **Affect encoder** | Trained | Test VAD MAE **0.160** | Text maps to lexicon geometry reasonably; not human-rated emotion accuracy |
| **SNN amygdala** | Trained | Train MSE **0.028** (500 samples) | Spike path reconstructs labels; no independent valid/test SNN eval yet |
| **Affective gate** | Trained | Loss **14.63** (100 samples) | Proxy empathy log-prob improved under training; behavioral empathy not yet benchmarked |
| **Phase tests A–E** | 78 passed, 6 skipped | pytest | Unit/integration coverage for dynamics, coupling, hooks, checkpoints |

All three checkpoints live on Modal volume `saa-models`:

```
/models/affect/encoder.pt
/models/snn/amygdala.pt
/models/gate/affect_gate.pt
```

---

## What was trained

### Supervision signal (not ground-truth emotion)

Training does **not** use human continuous affect ratings or fMRI. Each EmpatheticDialogues
sample has a **discrete emotion label** (e.g. `anxious`, `devastated`, `grateful`). That label
maps to a **fixed 32-dimensional prototype** built from:

1. **VAD** (valence, arousal, dominance) per label — hand-specified in `emotion_lexicon.py`
2. **Macro-bucket geometry** filling the remaining dimensions

The models learn to predict these **lexicon targets**, not to "read minds."

| Model | Input | Output | Loss |
|-------|-------|--------|------|
| **Encoder** | Utterance text (`user: {prompt}`) | 32-d vector | MSE to lexicon target + contrastive (same-emotion cluster) |
| **SNN** | Encoder spike trajectory | 32-d vector | MSE to same lexicon target |
| **Gate** | 32-d affect vector + frozen Llama | Hidden-state bias | −mean log p(empathy tokens) at last prompt token |

**Llama weights are frozen** (4-bit NF4). Only the encoder MLP head, SNN weights, and gate
projection are updated.

### Architecture (inference path)

```
Transcript messages
  → Hybrid encoder (frozen MiniLM-L6-v2 + trainable MLP head)
  → 32-d affect trajectory per turn
  → normalize_affective
  → delta_modulate (spike threshold θ = DELTA_THETA)
  → LIF amygdala (LIFAmygdala)
  → AffectDynamics (chat: leaky integrator across turns)
  → AffectiveGate → forward hooks on last Llama decoder layers
  → modulated generation
```

**Neutral policy (AF-4):** chat uses **hooks-off** for neutral, not a zero vector through a
trained bias. Gate bias is zeroed after each training step so `gate(0) ≈ 0` at checkpoint save.

---

## Data

| Item | Value |
|------|-------|
| Corpus | [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (CC-BY-NC) |
| Download | `scripts/download_empatheticdialogues.py` → local; Modal auto-download to `/models/data/raw/empatheticdialogues` |
| Train / valid / test | **17,844** / **2,763** / **2,542** |
| Holdout leaks filtered | **0** (train and valid) |
| Encoder backend | `hybrid` (sentence-transformers/all-MiniLM-L6-v2 + MLP) |

---

## Training runs

### Commands used

```powershell
# Full M1 (encoder + SNN; gate run separately)
modal run train_m1.py --encoder-epochs 2 --snn-samples 500 --snn-epochs 2 --skip-gate

# Gate (after encoder + SNN on volume)
modal run train_gate.py --max-samples 100 --epochs 1
```

Modal apps:

- Encoder + SNN: [ap-hEHgHm6urHNRMEWFQoSWmY](https://modal.com/apps/hrudayiitb/main/ap-hEHgHm6urHNRMEWFQoSWmY)
- Gate: [ap-sXqVveTlvzlQ0o3AhoLI6T](https://modal.com/apps/hrudayiitb/main/ap-sXqVveTlvzlQ0o3AhoLI6T)

---

## Component results

### 1. Affect encoder (`/models/affect/encoder.pt`)

**Config:** 2 epochs, full train split, batch size 64, Adam lr=1e-3, contrastive weight 0.25.

| Metric | Epoch 0 | Epoch 1 | Test |
|--------|---------|---------|------|
| Train MSE | 0.000284 | 0.000240 | — |
| Valid MSE | 0.01676 | 0.01637 | — |
| Valid VAD MAE | 0.1620 | 0.1630 | — |
| **Test VAD MAE** | — | — | **0.1601** |

**Artifacts:** `train_encoder.json`, `emotion_lexicon.json` (on volume under `/models/affect/`).

#### Interpretation

- **Test VAD MAE ≈ 0.16** is measured on the **first three dimensions** of **normalized** 32-d
  vectors vs lexicon VAD targets (roughly in [−1, 1]).
- This means valence/arousal/dominance are often directionally correct but **not precise** —
  expect ~0.16 average error per VAD axis, not fine-grained emotion classification.
- Low train MSE (≈3×10⁻⁴) vs higher valid MSE (≈0.016) suggests the head fits training utterances
  tightly; valid/test VAD MAE is the more meaningful generalization signal here.
- **This is not "emotion accuracy %."** Labels are coarse and targets are synthetic prototypes.

---

### 2. SNN amygdala (`/models/snn/amygdala.pt`)

**Config:** 500 train samples (subset), 2 epochs, Adam lr=1e-3, encoder checkpoint loaded
(`encoder_source: trained`).

| Metric | Epoch 0 | Epoch 1 |
|--------|---------|---------|
| Train MSE | 0.0356 | **0.0281** |

**Artifacts:** `benchmarks/train_snn.json` on volume.

#### Interpretation

- MSE decreased ~21% over one epoch on the **same 500 samples** — learning occurred.
- No held-out SNN metric was logged; this is **training-set reconstruction** only.
- The SNN is trained to map **encoder-derived spikes → lexicon 32-d**, same target family as
  the encoder. Whether the SNN adds information beyond the encoder's last vector requires an
  **ablation** (encoder-only vs encoder+SNN) on Phase 4 or chat A/B — not yet run post-training.

---

### 3. Affective gate (`/models/gate/affect_gate.pt`)

**Config:** 100 distress-biased samples (`anxious`, `sad`, `afraid`, `terrified`, `devastated`,
`distress`), 1 epoch, frozen W4 `meta-llama/Llama-3.2-1B-Instruct`, Adam lr=1e-4.

| Metric | Value |
|--------|-------|
| Samples | 100 |
| Final loss | **14.628** |
| Empathy token set | sorry, understand, here, support, feel, help, care |

**Artifacts:** `train_gate.json` on volume under `/models/gate/`.

#### Interpretation

- Loss = **negative mean log-probability** of empathy-related tokens at the **last prompt
  token** with affect hooks active. **Lower is better**; there is no pre-training baseline
  logged for comparison.
- A loss of ~14.6 on a large vocabulary implies empathy tokens were **not** dominant mass at
  that position after one epoch — modest nudge, not convergence.
- This does **not** prove replies are more empathetic to humans; lexical empathy score and
  logit-KL vs neutral (Phase 4 extended) are the right behavioral validators.
- **AF-4 fix applied:** `gate.proj.bias` zeroed after each optimizer step so saved checkpoints
  satisfy `assert_gate_noop` (‖gate(0)‖ < ε).

---

## Issues encountered and fixes

| # | Failure | Root cause | Fix |
|---|---------|------------|-----|
| 1 | Local training missing `sentence_transformers` | Hybrid encoder default on Windows CPU | Migrated real training to **Modal** (`affective_image` includes deps) |
| 2 | `ModuleNotFoundError: train_affect_encoder` on Modal worker | Root-level train scripts not in container | Moved loops to `src/train/`; Modal functions in `src/train/modal_jobs.py` |
| 3 | `ModuleNotFoundError: sentence_transformers` in **gate** job | `train_gate_remote` used base `image` not `affective_image` | Gate function now uses `affective_image` |
| 4 | `Inference tensors cannot be saved for backward` (encoder) | MiniLM `encode()` under inference mode | `.clone()` embeddings before MLP backward |
| 5 | CUDA/CPU mismatch (encoder) | MiniLM on GPU, MLP on CPU | `model.to(device)` in training loop |
| 6 | `Gate(0) norm exceeds eps` at save | Bias drift during gate training | Zero bias after each `opt.step()` |
| 7 | EmpatheticDialogues path on Modal | `EMPATHETICDIALOGUES_DIR` pointed at local repo path | `src/runtime_paths.py` + `ensure_empatheticdialogues()` on volume |

---

## Test suite (Tracks A–E)

Post-training pytest (excluding slow hybrid encoder test):

```
78 passed, 6 skipped, 0 failed
```

**Extended coverage:**

| Track | File | What was verified |
|-------|------|-------------------|
| A | `test_dynamics.py`, `test_chat.py` | Decay, reset, multi-turn affect evolution via mocked `ChatEngine` |
| B | `test_coupling.py`, `test_affect_metrics.py` | Coupling monotonicity; coupling correlation on identical trajectories |
| C | `test_stdp.py` | STDP LTP and weight clamp (exploratory) |
| D | `test_hooks.py` | Zero-affect noop, strength scaling, `None` affect skips hooks |
| I/M1 | `test_checkpoints.py`, `test_m1_*` | Checkpoint round-trip, fixture SNN train, engine encoder path |

**Expected skips (unimplemented modules):** `test_microscope_api`, `test_phenotype`,
`test_profiles`, `test_brain_alignment`, `test_prosody`; `test_gate_training` on CPU without CUDA.

---

## What we can and cannot claim

### Supported claims

- v1 stack is **trained on labeled empathetic dialogue**, not TRIBEv2 predictions.
- Three checkpoints form a **closed pipeline**: text → 32-d → spikes → SNN → Llama hooks.
- Encoder generalizes to held-out **test** split for VAD MAE (~0.16).
- Gate training preserves **neutral = hooks-off** invariant at checkpoint save.
- Unit tests guard dynamics, coupling, hook noop, and checkpoint I/O.

### Unsupported claims (yet)

- "The model **accurately detects** user emotion" — targets are lexicon prototypes, not ratings.
- "The SNN **improves** generation quality" — no post-M1 Phase 4 / chat A/B with new checkpoints.
- "Gate training **makes** replies empathetic" — only empathy-token log-prob proxy, 100 samples, 1 epoch.
- "Brain-aligned" or "amygdala-like" — brain-inspired architecture only; no fMRI alignment (v2).

---

## Metric glossary

| Metric | Definition | Good direction |
|--------|------------|----------------|
| **VAD MAE** | Mean absolute error on valence, arousal, dominance (dims 0–2 after normalize) | Lower |
| **MSE (32-d)** | Mean squared error vs full lexicon target vector | Lower |
| **SNN train MSE** | MSE between amygdala output and lexicon target on spike input | Lower (train only logged) |
| **Gate loss** | −mean log p(empathy token ids) at last prompt token | Lower |
| **tribev2_used** | Always `false` for this session | — |

---

## Recommended next validation

See **[verification_session.md](./verification_session.md)** for the full post-M1 verification log (pytest, Modal Phase 4/loop, audit).

1. **Phase 4 extended** (`benchmark_phase4_extended.py`) with loaded M1 checkpoints — logit KL,
   empathy lexical delta, strength sweep; neutral = hooks-off.
2. **Chat A/B** (`benchmark_phase_chat_ab.py`) — distress vs neutral transcripts.
3. **Holdout scenarios** (`benchmark_phase_scenarios.py`) — hooks-off vs transcript-conditioned affect on `data/scenarios/`.
4. **Encoder vs encoder+SNN ablation** — does amygdala change metrics beyond encoder last vector?
5. **Longer gate training** — more samples, multiple epochs, log baseline loss and lexical empathy
   before/after on a fixed holdout prompt set.

---

## Holdout scenario eval (synthetic scripts)

**Added:** 29 template-generated multi-turn scripts in `data/scenarios/` (10 distress→recovery, 7 conflict, 6 factual neutral, 6 tone shift). All utterances are registered in `collect_holdout_texts()` and filtered from encoder/gate training splits. Regenerate with:

```powershell
$env:PYTHONPATH="."
py -3 scripts/generate_scenarios.py
```

| Category | Count | Purpose |
|----------|-------|---------|
| `distress_recovery` | 10 | Multi-turn arcs with partial recovery |
| `conflict` | 7 | Escalation / blame / betrayal |
| `factual_neutral` | 6 | Hooks-off A/B — affect should not derail facts |
| `tone_shift` | 6 | Sudden valence flip mid-conversation |

**Benchmark:** `benchmark_phase_scenarios.py` compares **hooks-off** (AF-4 neutral) vs **hooks-on** with encoder→SNN affect from the full transcript, then reports `text_changed`, empathy lexical delta, and tone metadata. Default run evaluates 8 representative scenarios; pin two with:

```powershell
modal run benchmark_phase_scenarios.py --scenario-ids tone_calm_to_panic,factual_week_planning
```

### Highlighted scenarios (initial eval)

Two scenarios were exercised locally (encoder-only, hash backend — exploratory, not scientific):

| Scenario | Category | Turns | Encoder tone (hash) | Significance |
|----------|----------|-------|---------------------|--------------|
| `tone_calm_to_panic` | tone_shift | 6 | calm | Sudden reply-all panic; tests whether affect path responds to whiplash without TRIBEv2 |
| `factual_week_planning` | factual_neutral | 5 | calm | Tagged `hooks_off_ab`; factual content should stay stable under hooks-off baseline |

**Local artifact:** `data/artifacts/scenario_encoder_eval.json`

**Interpretation (encoder-only):** Hash backend cannot distinguish panic vs planning (both `calm`) — expected. **Scientific behavioral claims require Modal** run with trained hybrid encoder + gate on the same scenario IDs. Success criteria for the full benchmark:

- **Tone-shift scenarios:** `text_changed=true` and empathy_delta ≥ 0 vs hooks-off on ≥50% of distress/tone_shift cases.
- **Factual neutral:** `text_changed=false` or minimal logit KL on ≥80% — affect should not rewrite factual answers.
- **Do not claim** empathy quality from lexical heuristics alone (`metric_type: heuristic`).

### Chat engine AF-4 (completed this session)

`ChatEngine` no longer registers persistent hooks at init. Hooks attach **only during `generate_reply`** when `hook_strength > 0` and `‖affect_vector‖ > GATE_NOOP_EPS`. Neutral / zero-affect turns run hooks-off, matching Phase 4 benchmark policy.

### Verification follow-ups (status)

| Step | Status |
|------|--------|
| Holdout scenarios + eval benchmark | Done (29 scripts, `benchmark_phase_scenarios.py`) |
| Chat engine AF-4 hooks-off neutral | Done |
| Loop benchmark ≥8-turn arcs + `exploratory` flag | Done (`benchmark_phase_loop.py`) |
| Gate v2 (contrastive + anti-collapse loss) | **Superseded** — see Gate v3 |
| Gate v3 (listener CE + SNN-aligned) | **Done** — code in `src/train/gate.py` |
| M2 SNN membrane carryover | **Done** — `lif_network` + session/engine threading |
| M3 Microscope API | **Done** — `src/serve/microscope_api.py`, `run_microscope.py` |

---

## Behavioral verification (post gate re-train)

**Gate re-train:** `py -3 -m modal run train_gate.py --max-samples 500 --epochs 3`  
Defaults updated: `GATE_TRAIN_MAX_SAMPLES=500`, `GATE_TRAIN_EPOCHS=3`, `GATE_GPU_TIMEOUT_SEC=7200`.

**Training loss** (`data/artifacts/modal/train_gate.json`): 500 samples · epoch losses **10.34 → 4.67 → 3.73** (vs 14.63 on 100×1 run).

**Checkpoints at eval time:** `encoder_source`, `amygdala_source`, `gate_source` all **`trained`** on Modal volume `saa-models`.

### Phase 4 extended (valid short-form claims)

Modal app: `ap-YFsxuWdkxDPKXOmA3nuSjs` · Artifact: `/models/benchmarks/phase4_extended.json`

| Metric | Before (100 samples, 1 epoch) | After (500 samples, 3 epochs) |
|--------|-------------------------------|-------------------------------|
| Text changed (hooks-off vs high) | 4/5 (80%) | **5/5 (100%)** |
| Mean logit KL | 0.0085 | **0.312** |
| Phenotype empathy_delta (heuristic) | −0.008 | **+0.037** |
| `conflict_deescalation` empathy_delta | — | **+0.089** |

**Supported claim:** At **64 tokens**, fixed high-affect vector + trained gate **measurably shifts** Llama outputs (logit KL and text diffs). Hooks-off neutral baseline is honest (AF-4).

### Holdout scenarios (transcript-conditioned — collapse)

Modal app: `ap-0WKcIy29Egyng9qSKb4eUF` · `scientific: true` · `max_new_tokens: 64`

| Scenario | hooks-off | hooks-on (transcript affect) | Verdict |
|----------|-----------|------------------------------|---------|
| `tone_calm_to_panic` | Coherent advice | **`feelfeel…` repetition** | Modulation yes; quality **failed** |
| `factual_week_planning` | Coherent planning | **`feelfeel…` repetition** | Factual stability **failed** when hooks on |

Lexical `empathy_delta` spikes (+0.83 mean) are **misleading** — driven by token repetition, not better responses.

### Chat A/B (transcript-conditioned — partial collapse)

Modal app: `ap-Wopabt4psH0O3ygwMsCpZ3` · `max_new_tokens: 64`

| Arc | Dominant tone | Generation quality |
|-----|---------------|-------------------|
| **neutral** | warm | Coherent planning advice |
| **distress** | tense | Collapses to `sorry`/`feel` repetition |
| **hopeful** | tense | Starts coherent, then `feel sorry` loop |

Encoder path correctly separates tones (`warm` vs `tense`); **gate overfits empathy-token logprob** under transcript-derived affect on multi-turn prompts.

### Significance (what we can publish)

| Claim | Status |
|-------|--------|
| Trained stack loads and modulates short single-turn prompts (Phase 4) | **Supported** |
| Hooks-off = true neutral (AF-4) | **Supported** |
| Holdout scenarios never in training | **Supported** |
| Transcript-conditioned chat quality improves with affect | **Not supported** — mode collapse |
| Lexical empathy_delta alone | **Do not use** — inflates on repetition |

### Gate v3 — listener CE (real collapse fix)

**Implemented (2026-07-03):** Teacher-forced CE on human **listener replies** from EmpatheticDialogues with hooks-on vs hooks-off margin (distress) / noop (neutral). Affect vectors built via **encoder → SNN → clip** (same path as chat). Balanced distress/neutral batches. Holdout every 50 steps with `collapse_score` ranking.

Re-run on Modal:

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3 -m modal run train_gate.py --max-samples 500 --epochs 2
py -3 scripts/run_behavioral_verification.py --skip-train
```

**Checkpoint tag:** `gate_version: v3_listener_ce` in `train_gate.json` (bumped to
`v3.1_listener_ce_hardened` after the checkpoint-selection/loss-leak fixes below —
re-train to get a v3.1-tagged checkpoint).

**Success criteria (unchanged):**

- Phase 4: `fraction_text_changed` ≥ 0.8, mean KL > 0.05, phenotype empathy_delta ≥ 0.
- Chat A/B distress: no empathy-token run-length > 8; coherent reply preview.
- Scenarios: `factual_week_planning` hooks-on may change text but `collapse_detected: false`.

**Prior gates:** v1 (empathy-ID last token) and v2 (contrastive margin) both collapsed on multi-turn holdout — see tables above. Post–v3 numbers belong in `data/artifacts/behavioral_verification.json` after Modal re-run.

~~Gate v2 contrastive objective~~ **Superseded by v3 listener CE.**

### Gate v3.1 — hardening after collapse-free training run (2026-07-03)

A 500×2 Modal run of Gate v3 completed with `collapse_score: 0.0` at every holdout
checkpoint from step 550 onward and no early-stop. Before trusting that result, a
`business-logic-auditor` review of `src/train/gate.py`, `gate_loss.py`,
`gate_vector.py`, and `gate_holdout.py` was run and found several ways a
collapse-free holdout log could still be misleading. Fixes applied:

| # | Finding | Fix |
|---|---------|-----|
| 1 | Best-checkpoint selection used strict `holdout_score < best`, so the *first* step to hit the eventual floor score (e.g. step 50) was permanently kept — every later tie (including the fully-trained final step) was silently discarded, and the winning step was never recorded. | `_is_new_best_checkpoint` now uses `<=` so ties favor the later (more-trained) checkpoint; `train_gate.json` now records `best_step` and `total_steps`. |
| 2 | Nothing distinguished a genuinely helpful, affect-conditioned gate from one that collapsed to a trivial near-inert mapping — both would produce identical "0.0 collapse" holdout logs. | `_eval_holdout_gate` now also generates a hooks-off baseline per prompt and records `hooks_off_preview`, `text_changed`, and `gate_output_norm` (‖gate(aff_high)‖) so inertness is directly observable in `holdout_eval.json`. |
| 3 | The neutral-bucket loss included an unconditional `ce_on` term identical to the distress bucket, directly rewarding hooks-on for generically improving prediction of the neutral listener reply — the opposite of the "hooks are inert on neutral input" invariant `neutral_noop_loss` was meant to encode. | Neutral bucket now trains on `neutral_noop_loss` alone (no bare `ce_on` term); distress bucket keeps `ce_on` since behavior-cloning the listener reply *is* the intended distress objective. Regression tests: `test_neutral_loss_does_not_reward_lower_ce_on`, `test_distress_loss_rewards_lower_ce_on`. |
| 4 | `gate_noop_regularizer` (`GATE_NOOP_REG_WEIGHT * ‖gate(0)‖`) is mathematically dead: `gate(0) = W·0 = 0` for *any* `W`, and `d(Wx)/dW at x=0` is the zero tensor, so it always evaluated to exactly 0 with exactly 0 gradient regardless of the gate's weights. It contributed nothing. | Removed from the loss and from `gate_loss.py`. `assert_gate_noop` (a post-hoc structural check, not a trained objective) is unaffected and still passes. |
| 5 | Llama's non-quantized parameters (embeddings, layernorms, `lm_head`) were never explicitly frozen before training; every `loss.backward()` through the hooked forward pass accumulated unused gradients on them for the whole run, wasting VRAM. | Added `_load_frozen_llama()`, which sets `requires_grad = False` on every Llama parameter right after loading, before the gate/optimizer are constructed. |
| 6 | Gate training-time checkpoint selection (`holdout_prompts()`) reused 2 of the 5 `PHASE4_ABLATION_PROMPTS` — the same prompts later used to claim independent Phase 4 validation. Any Phase 4 report on those 2 prompt IDs for this checkpoint would be circular, not independent. | Added a disjoint `GATE_TRAIN_HOLDOUT_PROMPTS` (distinct wording, distinct ids) in `src/config.py`; `holdout_prompts()` now reads from it. `PHASE4_ABLATION_PROMPTS` stays untouched by gate training. |
| 7 | Holdout eval generated only 64 tokens — shorter than the v1/v2 failure mode, which tended to emerge later in generation, and shorter than real chat (`CHAT_MAX_NEW_TOKENS=256`). | Bumped to `GATE_HOLDOUT_MAX_NEW_TOKENS=96` (still short for compute cost reasons — full-length Phase 4 / chat A-B runs remain the real long-generation check). |

**Not fixed, flagged as a residual limitation:** `listener_sequence_ce` trains hooks-on with teacher forcing on the *gold* human listener reply, but real generation conditions on the model's own sampled tokens — a standard exposure-bias gap. Holdout eval (which does use the model's own greedy rollout) is the practical mitigation; this is why holdout/Phase 4/chat A-B results, not training loss, are the trustworthy signal for "did collapse actually go away."

**Verdict:** the "no collapse in training holdout" result from the 500×2 run should be re-verified with these fixes (a re-trained checkpoint will report `best_step`/`gate_output_norm`/`text_changed` so undertraining and inertness are directly checkable), and independently confirmed via `scripts/run_behavioral_verification.py` (Phase 4 extended, 29 scenarios, chat A/B) before being treated as evidence the collapse problem is fixed.

### Collapse-detector false positive found during v3.1 behavioral verification (2026-07-04)

The v3.1 checkpoint above was retrained (500×2, `best_step`/`gate_output_norm`/`text_changed`
all recorded as intended) and behavioral verification ran successfully: Phase 4 extended
showed `fraction_text_changed: 1.0`, and scenario holdout showed `collapse_detected: false`
with meaningful `text_changed`/empathy deltas — real, working modulation, not inertness.
But the **Chat A/B** benchmark reported `collapse_detected: true` for the "distress" and
"hopeful" scenarios even though `reply_preview` read as a coherent, on-topic empathetic
response with no visible repetition. Investigation found two compounding bugs in the
detector itself, not in the gate or training:

| # | Finding | Fix |
|---|---------|-----|
| 1 | `tokenizer.decode(out[0], ...)` in `generate_text` decodes the *full* sequence (prompt + chat template + generation), and every caller ran `detect_empathy_collapse`/`collapse_score` on that full string. In `benchmark_phase_chat_ab.py`, the "distress"/"hopeful" scenarios' scripted transcript history literally contains "...I feel awful...", "...I'm here with you..." — words from the benchmark author's own dialogue, not the model, were being scored as if the model produced them. `_eval_holdout_gate` (`src/train/gate.py`) and `benchmark_phase_scenarios.py` had the same pattern (lower risk there since holdout prompts are single-turn, but still incorrect). | `generate_text` now also returns `stats["new_text"]` — the newly generated continuation only (tokens after the prompt length), decoded separately. All three call sites (`_eval_holdout_gate`, `benchmark_phase_chat_ab.py`, `benchmark_phase_scenarios.py`) now run collapse detection on `stats["new_text"]` instead of the full prompt+generation text. |
| 2 | `detect_glued_empathy_morphs`'s low-diversity fallback compared `len(set(compact))` (bounded by the 26-letter alphabet, so at most 26) against `len(compact) // 4`. For any text longer than ~104 characters this is *always* true, so the fallback silently degenerated into a bare `hits >= 6` substring count — meaning any sufficiently long, normal-length response mentioning "feel"/"care"/"here"/"under"/"sorry" six times total (easy for a genuinely empathetic multi-sentence reply, or a long prompt+reply blob) would false-positive regardless of whether those words were glued together or scattered naturally across separate sentences. | Replaced with a length-normalized coverage ratio: `covered_chars / len(compact) >= 0.35` (in addition to `hits >= 6`), so the check only fires when these morphemes actually dominate the text — the real signature of glued-repetition collapse — not merely appear in it. |

Verified against the exact reproduction of the false-positive case (chat-template + scripted
distress transcript + a genuine multi-sentence empathetic reply): `detect_empathy_collapse`
went from `True` (false positive, `collapse_score=0.45`) to `False` (`collapse_score=0.0`)
after the fix, while all existing collapse-regression tests (v1 sorry-loop, v2 glued-morph
chain, mixed empathy word-salad, low-diversity non-glued repeat) still correctly return
`True`. New regression tests added to `tests/test_gate_holdout.py` and a new
`tests/test_llm_loader.py`. Full suite: 127 passed, 4 skipped (CUDA-only).

**Re-verified (2026-07-04):** re-ran `benchmark_phase_chat_ab.py` on Modal against the
existing v3.1 checkpoint (no retraining needed — this was a scoring bug, not a model/training
bug). All three scenarios now correctly report `collapse_detected: false`, with the
newly-generated text for each reading as coherent, on-topic, and genuinely empathetic:

- **distress**: *"It's normal to feel that way after a tough test... 1. Allow yourself to grieve... 2. Take care of yourself: Make sure you get enough sleep tonight..."*
- **neutral**: *"Considering you have work tasks and a gym session planned for the day, here's a suggested plan: 1. Work task..."*
- **hopeful**: *"Feeling hopeful is a great step! Now that you've got some time to think... 1. Take care of yourself: Make sure you get plenty of rest, eat well, and stay hydrated..."*

This also confirms the gate isn't just avoiding collapse by going inert: both distress and
hopeful show `text_changed: true` vs. the neutral baseline with meaningful empathy deltas
(+0.2551 distress, +0.2417 hopeful) and correctly differentiated sentiment (distress `0.0`,
hopeful `1.0`). Gate v3.1 + the collapse-detector fix are both confirmed working as intended
across all three behavioral verification benchmarks (Phase 4 extended, scenario holdout,
chat A/B).

---

## Changelog

| Date | Event |
|------|-------|
| 2026-06-20 | M1 encoder + SNN trained on Modal (2 epochs / 500 samples) |
| 2026-06-22 | Gate trained on Modal (100 samples, 1 epoch); AF-4 bias fix applied |
| 2026-06-22 | Phase tests A–E extended; 78 passed |
| 2026-06-22 | 29 holdout scenarios + scenario eval benchmark; ChatEngine per-generation hooks (AF-4) |
| 2026-06-22 | Gate re-train 500×3 epochs; Phase 4 verified; scenario/chat expose empathy-token collapse |
| 2026-06-20 | Gate v2 objective + holdout early-stop; M2 membrane carryover; M3 Microscope API |
| 2026-07-03 | Gate v3 listener CE + SNN-aligned training; balanced distress/neutral batches |
| 2026-07-03 | Gate v3.1 hardening after audit: fixed best-checkpoint tie-break, removed dead noop regularizer, fixed neutral-bucket loss leak, froze Llama params, disjoint gate-training holdout prompts, added gate_output_norm/text_changed diagnostics |
| 2026-07-04 | v3.1 Modal retrain (500×2) completed; behavioral verification found chat A/B false-positive `collapse_detected` caused by scoring full prompt+reply text and a broken diversity fallback in `detect_glued_empathy_morphs` — both fixed, new regression tests added |
| 2026-07-04 | Chat A/B re-run against existing v3.1 checkpoint post-fix: all 3 scenarios `collapse_detected: false`, coherent empathetic replies, real (non-inert) affect modulation confirmed |
