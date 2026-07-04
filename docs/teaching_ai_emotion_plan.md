# Teaching AI Emotion — File-Level Implementation Plan

> Goal: evolve the Spiking Affective Adapter from "inject an affect vector and benchmark it"
> into a system that **processes emotion as a temporal, causal, trainable, and observable**
> internal state — and lets us *see* its effect on a frozen LLM.
>
> Audience: this doc is meant to be public (GitHub profile). It must be honest about what is
> real (spiking neuromodulation, behavioral supervision) vs heuristic (lexical/tone proxies) vs
> optional (fMRI alignment in v2). **TRIBEv2 is no longer the primary training signal** — see §1.

---

## 0. Current-State Findings (grounding — read before planning)

These were established by reading the actual source, not the README. They are the foundation
for the tracks below. Three are latent correctness gaps that must be fixed for the
"emotion" claim to be defensible.


| #   | Finding                                                                                                   | File / evidence                                                                                        | Impact on "teaching emotion"                                                               |
| --- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| F1  | Affect only refreshes every `AFFECT_REFRESH_SEC = 300`s; within that window affect is frozen.             | `src/config.py:51`, `src/chat/engine.py:96`, `session.needs_affect_refresh`                            | Emotion behaves like a static knob, not a process. **Track A.**                            |
| F2  | `LIFAmygdala` and `AffectiveGate` are randomly initialized in chat and never loaded from `amygdala.pt`.   | `engine.py:50,157-160` (lazy `LIFAmygdala(...)`, no `load_state_dict`); gate created at `engine.py:50` | Modulation is untrained noise. "AI feels" is not yet true. **Track I (prereq).**           |
| F3  | SNN calls `init_leaky()` every forward → membrane state resets each turn.                                 | `src/brain/lif_network.py:49-50`                                                                       | No emotional inertia / carryover. **Track A + D.**                                         |
| F4  | Trait scalars and tone are explicitly display-only heuristics.                                            | `signatures.compute_traits`, `tone_markers.dominant_tone`                                              | Cannot be used as scientific validation; keep as UI only. **Track C honesty.**             |
| F5  | TRIBEv2 silently falls back to synthetic; `source` string is the only signal.                             | `tribev2_client.run_tribev2_predict:105-107`                                                           | Any new path must surface `source` to UI/artifacts. **Cross-cutting invariant.**           |
| F6  | `generate_reply` trims messages in-place, but affect refresh uses full transcript at a different cadence. | `engine.py:99-105`, `refresh_affect:142`                                                               | Affect can be computed on different text than generation sees. **Track A must reconcile.** |


### Invariants every track must preserve

- `AFFECT_DIM = 32` end-to-end (encoder/labels→compress→SNN→gate→state).
- v1 training supervised by **labeled dialogue** (§1), not TRIBEv2 predictions.
- Quantization separation: weights = NF4 (Phase 1a); KV = INT8/INT4 or FP8 (Phase 1b). Never conflate.
- `source` (real TRIBEv2 vs synthetic) must be propagated to every artifact and UI surface.
- A/B and sweep comparisons differ only in the affect vector (same prompt, temp, tokens, strength, seed).
- Hooks registered before generate, removed in `finally`/`cleanup`.
- Heuristic metrics (lexical empathy, tone) are relative proxies, never ground-truth emotion.

---

## 1. Data & Supervision Strategy (post-TRIBEv2)

> **Decision:** TRIBEv2 cortical predictions (and even subcortical amygdala channels) are **not**
> suitable as training ground truth for an "amygdala that learns emotion." Default
> `facebook/tribev2` is cortical-surface-only; our `AffectiveCompressor` uses index-based vertex
> chunks, not anatomical ROIs; subcortical TRIBEv2 scores are 2–3× lower than cortical; and training
> on model-predicted BOLD compounds error. See discussion in project chat (Jun 2025).
>
> **New primary signal:** labeled emotional language → 32-d affect state → SNN dynamics → gate
> training with **LLM behavioral targets**. TRIBEv2 becomes an **optional demo / v2 alignment
> channel**, never the supervisor.

### 1.1 What we are teaching (supervision targets)


| Component                                 | Supervised by                                          | Loss / objective                                                                                          | Validated by                                                                          |
| ----------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Text → 32-d encoder** (`AffectEncoder`) | Emotion labels on dialogue text (VAD + category)       | MSE / contrastive to label-derived 32-d prototype                                                         | Held-out label reconstruction; transcript→vector monotonicity on valence              |
| **SNN (`LIFAmygdala`)**                   | Spike trains from encoder output + temporal continuity | Next-step prediction + membrane regularizer; optional STDP from spike timing (Track B)                    | Stable firing stats; trajectory smoothness; no unbounded drift (AF-11)                |
| **Gate (`AffectiveGate`)**                | Same transcripts + frozen Llama                        | Contrastive log-prob / lexical empathy delta under high vs neutral affect; **neutral = hooks-off** (AF-4) | Phase 4 extended metrics; `test_gate_training.py`                                     |
| **TRIBEv2 (optional)**                    | Nothing in v1                                          | Inference-only feature channel if installed                                                               | Correlation with text-VAD on held-out stimuli; label `scientific: false` if synthetic |


**Public GitHub framing (v1):** *"Brain-inspired spiking affective adapter trained on emotional dialogue
and validated by measurable shifts in a frozen Llama — not a digital clone of the human amygdala."*

### 1.2 v1 corpus — concrete datasets

#### A. EmpatheticDialogues (primary external corpus)

- **Source:** [facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (~25k conversations, 32 emotion labels + speaker/listener turns).
- **License:** CC-BY-NC (same constraint as TRIBEv2 — non-commercial research/demo OK).
- **Why:** Real human-labeled emotional language at scale; each conversation anchored to a situation
emotion (e.g. *anxious*, *grateful*, *angry*) with empathic responses — directly matches "teach AI
emotion from language."
- **On-disk layout (not checked in — download at train time):**

```
data/raw/empatheticdialogues/
  train.csv
  valid.csv
  test.csv
```

- **Fields used:** `conv_id`, `utterance`, `context` (emotion category), `prompt` (situation text),
`speaker_idx`, `utterance_idx`.
- **Preprocessing:** build multi-turn transcripts per `conv_id`; map 32 EmpatheticDialogues emotions →
**VAD triple** via a fixed lookup table (`src/affective/emotion_lexicon.py`); optionally add arousal
from utterance length / punctuation heuristics.
- **Split:** official train/valid/test from the dataset; **never** mix benchmark holdouts (below) into
train.

#### B. In-repo benchmark holdouts (evaluation-only, small)

Fixed scenarios already in `src/config.py` — use for **eval and smoke tests only**, not training:


| Key                       | Location           | Role                                                         |
| ------------------------- | ------------------ | ------------------------------------------------------------ |
| `CHAT_AB_TRANSCRIPTS`     | `config.py:97-122` | distress / neutral / hopeful arcs — chat A/B + coupling eval |
| `PHASE4_ABLATION_PROMPTS` | `config.py:64-93`  | 5 single-turn prompts — strength sweep + phenotype card      |
| `BENCHMARK_PROMPT`        | `config.py:39-42`  | amygdala-explain sanity check                                |


Add ~20–30 **synthetic multi-turn scripts** in `data/scenarios/` (distress→recovery, conflict, factual neutral) generated from templates — supplements EmpatheticDialogues for edge cases (sudden tone shift, hooks-off A/B). These are **held out** from gate training.

#### C. Optional v1 augment (later within M1)


| Source                                                                                  | Size                              | Use                                                                |
| --------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | ~58k Reddit comments, 27 emotions | Single-turn VAD/category supervision for encoder pretrain          |
| Project chat logs (`phase_chat.json`)                                                   | Opt-in, local                     | STDP / calibration feedback (Track G) — never committed by default |


### 1.3 v1 pipeline — text labels → 32-d → spikes → train

Replace TRIBEv2-as-supervisor with this path for **all training and interactive chat**:

```
Utterance / transcript
  → AffectEncoder (small MLP or frozen MiniLM → 32-d)
  → normalize_affective (existing)
  → delta_modulate (DELTA_THETA, existing)
  → LIFAmygdala (SNN)
  → AffectDynamics (Track A, chat only)
  → AffectiveGate → Llama hooks
```

**New files (v1 data stack):**


| File                                      | Purpose                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `src/affective/emotion_lexicon.py`        | Map EmpatheticDialogues (+ GoEmotions) labels → VAD; build 32-d prototype vectors               |
| `src/affective/encoder.py`                | `AffectEncoder`: text → `(T, 32)` trajectory; trainable; replaces TRIBEv2 in chat/train paths   |
| `src/affective/dataset.py`                | `EmpatheticDialoguesDataset`, `ScenarioHoldoutDataset`; yields `(transcript, label_vad, split)` |
| `scripts/download_empatheticdialogues.py` | Fetch + verify checksums into `data/raw/`                                                       |
| `train_affect_encoder.py`                 | Modal/local: encoder pretrain on label reconstruction                                           |
| `train_snn.py` (rewrite)                  | SNN train on encoder spike outputs + temporal loss (not TRIBEv2 fMRI)                           |
| `src/brain/train_gate.py`                 | Gate contrastive train on frozen Llama + encoder vectors (Track I)                              |


**32-d label geometry (fixed, interpretable):**

- Dimensions 0–2: valence, arousal, dominance (VAD), scaled to [−1, 1].
- Dimensions 3–7: one-hot-ish cluster scores for 5 macro-buckets (distress, warmth, tension, calm,
neutral) derived from emotion lexicon.
- Dimensions 8–31: PCA filler trained on encoder hidden states **or** reserved zeros with L2 reg —
keeps `AFFECT_DIM = 32` invariant; document which dims are semantically anchored vs learned.

**Encoder training objective (v1):**

```python
# Pseudocode — train_affect_encoder.py
loss = mse(encoder(text), vad_to_32d(label)) + contrastive(same_emotion_pairs)
```

**Gate training objective (v1, Track I):**

```python
# Pseudocode — train_gate.py
# Neutral baseline = hooks OFF (AF-4), not zero vector
loss_high = -empathy_logprob(reply | prompt, affect=high_vec)  # distress transcripts
loss_neutral = kl(logits_hooks_off, logits_baseline)  # should be ~0
loss_reg = ||gate(zeros)||  # hard assert at save: < 1e-4
```

**Artifacts (Modal volume `saa-models`):**

```
/models/affect/
  encoder.pt
  emotion_lexicon.json
  train_encoder.json      # metrics, split sizes, label coverage
/models/snn/
  amygdala.pt             # now actually trained
/models/gate/
  affect_gate.pt
```

### 1.4 v1 train / eval protocol


| Split       | Contents                                                              | Used for                                                |
| ----------- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| **Train**   | EmpatheticDialogues train (~19k conv)                                 | Encoder + SNN + gate                                    |
| **Valid**   | EmpatheticDialogues valid                                             | Early stopping, hyperparams                             |
| **Test**    | EmpatheticDialogues test                                              | Report label reconstruction + gate behavioral delta     |
| **Holdout** | `CHAT_AB_TRANSCRIPTS` + `PHASE4_ABLATION_PROMPTS` + `data/scenarios/` | Never seen in train; Phase 4 ext + chat A/B + phenotype |


**Success criteria (v1 — publish on GitHub README):**

1. Encoder: held-out VAD MAE < 0.15 on test (macro-averaged).
2. Gate: on holdout distress prompts, empathy lexical delta > neutral hooks-off baseline (relative, not absolute truth).
3. Phase 4 ext: `fraction_text_changed` > 0 on ≥3/5 holdout prompts at strength=1.0.
4. Every artifact JSON includes `"supervision": "empatheticdialogues_v1"` and `"tribev2_used": false`.

### 1.5 TRIBEv2 demotion (what happens to existing code)


| Path                              | v1 role                                                                                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/affective/tribev2_client.py` | **Deprecated for training.** Keep for optional `--tribev2-demo` inference flag.                                                                       |
| `run_affective_pipeline.py`       | Move to `scripts/legacy/run_tribev2_demo.py` or mark deprecated in docstring.                                                                         |
| `refresh_affect()` in `engine.py` | Default: `AffectEncoder` on transcript; TRIBEv2 only if `use_tribev2=True` and package installed.                                                     |
| `source` field                    | New values: `encoder:empatheticdialogues`, `encoder:holdout`, `tribev2:...` (optional), `synthetic_fallback:...` (fail closed for scientific claims). |


Do **not** delete TRIBEv2 integration yet — it remains a optional "brain emulator demo" for v2 alignment
and for users who want to compare encoder vs TRIBEv2 side-by-side in the Microscope (Track E).

### 1.6 v2 corpus — optional fMRI alignment path

> **Goal:** add a *weak* neuroscience anchor without claiming TRIBEv2 ground truth. Real measured BOLD
>
> - segment emotion labels, aligned to the same 32-d space as v1.

#### Primary open dataset

- **OpenNeuro ds002322** — Alice in Wonderland narrative fMRI (multiple languages; English subset usable).
- **Reference pipeline:** [Emotional-Decoding](https://github.com/HansDahleKvadsheim/Emotional-Decoding) —
segment narrative (~8s) → LLM/human Plutchik labels → ROI or DFC features → classical decoders.

#### v2 additional files


| File                                    | Purpose                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------- |
| `src/affective/fmri_alignment.py`       | Load preprocessed ROI time series; align segment labels to TR                       |
| `scripts/preprocess_openneuro_alice.py` | Download ds002322; Schaefer-400 ROI extraction (document dependency on FSL/nilearn) |
| `train_affect_encoder.py` (extend)      | Add auxiliary loss: `λ * mse(encoder(text_segment), roi_to_32d(amyg_insula_acc))`   |
| `benchmark_phase_brain.py` (Track F)    | Report correlation between encoder 32-d and held-out ROI patterns — **not** TRIBEv2 |


#### v2 ROI targets (honest scope)


| ROI                       | Rationale                                                  | Expectation                                        |
| ------------------------- | ---------------------------------------------------------- | -------------------------------------------------- |
| Amygdala (Harvard–Oxford) | Core affect — but noisy at 3T, small                       | Weak correlation; report with confidence intervals |
| Insula + ACC              | Interoception / salience — better SNR for language emotion | Primary alignment target                           |
| Temporal language parcels | Narrative processing — strongest text–brain link in Alice  | Sanity check that encoder isn't random             |


**v2 success criteria:** encoder VAD predictions correlate with ROI-derived valence *above chance* on
held-out Alice segments (report r and p); if r < 0.15, publish negative result honestly and keep v1
behavioral story as primary.

#### v2 optional: TRIBEv2-subcortical as *feature*, not *label*

- Model: `facebook/tribev2-subcortical` (not default cortical `tribev2`).
- Extract **amygdala mask voxels only** via Harvard–Oxford — not index-based vertex slices.
- Use only as an **auxiliary input channel** at inference; compare against encoder in Microscope A/B pane.
- Never train SNN/gate to mimic TRIBEv2 output directly.

### 1.7 Milestone mapping (data work)


| Milestone               | Data deliverable                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| **M0 (new, before M1)** | Download EmpatheticDialogues; implement `dataset.py` + `emotion_lexicon.py`; holdout registry |
| **M1**                  | Train encoder + gate on v1 corpus; deprecate TRIBEv2-as-supervisor                            |
| **M5 (Track F)**        | Optional v2: preprocess Alice; alignment benchmark                                            |
| **M6**                  | User calibration logs (opt-in local only)                                                     |


**Minimum viable training path:** M0 + M1 only — no fMRI required to ship a defensible "teaching emotion" story.

### 1.8 Licenses & ethics (public repo)


| Asset               | License    | Constraint                                       |
| ------------------- | ---------- | ------------------------------------------------ |
| EmpatheticDialogues | CC-BY-NC   | Non-commercial; OK for GitHub research portfolio |
| GoEmotions          | Apache 2.0 | Optional augment — commercial-friendly           |
| OpenNeuro ds002322  | CC0        | v2 fMRI — cite dataset DOI in README             |
| TRIBEv2             | CC-BY-NC   | Optional demo only in v1                         |
| User chat logs      | N/A        | Opt-in, local, encrypted; never default upload   |


---

## Track I (PREREQUISITE): Make modulation real — train & load the gate + amygdala

Without this, every downstream "emotion" claim rests on random weights (F2). Do this first.
**All training data comes from §1 (EmpatheticDialogues + holdouts)** — not TRIBEv2.

### New files

- `src/brain/train_gate.py` — trains `AffectiveGate` (and optionally fine-tunes `LIFAmygdala`)
so that affect vectors produce *meaningful, directional* shifts in Llama logits rather than
random bias. Objective: contrastive — high-affect transcripts should increase target affective
lexical/log-prob signal vs neutral, with a regularizer keeping neutral ≈ no-op.
- `src/brain/checkpoints.py` — single source of truth for saving/loading `amygdala.pt` and a new
`affect_gate.pt`; defines a versioned `AffectCheckpoint` schema (`affect_dim`, `hidden_size`,
`model_id`, `git_sha`, `created_ts`).

### Modified files

- `src/chat/engine.py`
  - `__init__`: after creating `self.gate` and lazily creating `self.amygdala`, call
  `checkpoints.load_gate(self.gate)` and `checkpoints.load_amygdala(...)` if files exist;
  log `gate_source = "trained" | "random_init"` and expose it in `generate_reply` output.
  - Move `LIFAmygdala` instantiation out of `refresh_affect` (F2/F3) into `__init__` so the same
  trained instance persists for the whole session (fixes mid-session re-instantiation noted in
  auditor invariant 1.4).
- `train_snn.py` — rewrite to train on **encoder spike outputs** from §1 corpus (temporal + spike
losses), not TRIBEv2 fMRI; then call `train_gate.py`; write `amygdala.pt` and `affect_gate.pt` to
Modal volume with `model_volume.commit()`.
- `src/config.py` — add `GATE_CKPT_NAME = "affect_gate.pt"`, `AMYGDALA_CKPT_NAME = "amygdala.pt"`,
`AFFECT_HIDDEN_SIZE` resolved from model config at load (assert == checkpoint).

### Tests

- `tests/test_checkpoints.py` — round-trip save/load preserves shapes; refuses load on
`affect_dim`/`hidden_size` mismatch (fail closed, not silent).
- `tests/test_gate_training.py` — after a tiny training run, neutral vector ≈ no-op
(‖gate(0)‖ ≈ 0) and a fixed high-affect vector produces a reproducible non-zero shift.

### Risk

- Gate trained against the 1B model's `hidden_size`; loading against 3B would silently misalign.
Mitigation: checkpoint stores `model_id` + `hidden_size`; loader asserts.

### Revisions from audit (AF-2, AF-4, AF-10)

- **AF-2 (FATAL):** `train_snn.py` currently saves a *random* SNN (no affect loss). This track must
add a real supervised/contrastive amygdala objective in `train_gate.py` (joint or staged with the
gate). Until that exists, `amygdala.pt` must be **relabeled** "frozen untrained brain-derived
geometry" everywhere — never "trained / the AI feels." No public claim of a trained amygdala until
the objective lands and a held-out metric shows directional affect response.
- **AF-4 (INVARIANT BREAK):** a trained gate breaks the implicit `gate(0)=0` neutral. Redefine
**neutral = hooks-off pass** in `src/benchmark/hybrid_runner.py` and Phase-4 (added to this change's
file list), and add a **hard** assertion `‖gate(zeros)‖ < ε` in `checkpoints.save/load` (not just a
soft training regularizer).
- **AF-10:** decide STDP's role here — either wire `STDPUpdater` into Track B's closed loop or mark it
exploratory in `docs/benchmarks.md`; fix/document its `[0,1]` weight clamp. Add `tests/test_stdp.py`.

---

## Track A: Multi-turn affective state (emotion as a process)

Fix F1/F3/F6: affect should evolve **every turn** with decay + accumulation, and the SNN should
carry membrane state across turns (inertia), not reset.

### New files

- `src/affective/dynamics.py` — `AffectDynamics` dataclass: leaky integration of the per-turn
affect vector with `decay` (toward neutral) and `gain` (toward new evidence); exposes
`step(new_vec) -> state_vec` and `trajectory()` (list of states for visualization). This replaces
the wall-clock `AFFECT_REFRESH_SEC` gating for *interactive* chat (keep the 300s path for
long-idle re-grounding only).

### Modified files

- `src/brain/lif_network.py`
  - `LIFAmygdala.forward`: accept optional `mem1=None, mem2=None`; if provided, **do not** call
  `init_leaky()` — continue from passed membrane state. Return `(aff, stats, (mem1, mem2))`.
  - Add `reset_state()` and keep backward-compatible 2-tuple return via a `return_state=False` flag
  so existing benchmark callers (`extract_signature_from_pipeline`, `run_amygdala_on_spikes`) don't break.
- `src/chat/session.py`
  - Add fields: `affect_dynamics: AffectDynamics | None`, `affect_trajectory: list[list[float]]`,
  `snn_mem_state` (opaque carry), `turn_index: int`.
  - `to_log_dict`: include `affect_trajectory` and `turn_index` for the trajectory graph artifact.
- `src/chat/engine.py`
  - `generate_reply`: change ordering so affect updates **per turn** —
  `append(user)` → `refresh_affect(force=True, lightweight=True)` → `dynamics.step` →
  `_sync_affect_state` → `generate`. This satisfies auditor invariant 3 (refresh→sync→generate)
  on **every** turn, not every 300s.
  - Reconcile F6: compute affect on the **same trimmed `messages`** that generation will see
  (pass the trimmed list into `refresh_affect`), or explicitly document the asymmetry. Plan picks
  "compute on trimmed messages" for consistency.
  - `refresh_affect`: add `lightweight: bool` to skip full TRIBEv2 re-run when only the last turn
  changed (use incremental transcript), and thread persistent `snn_mem_state`.
- `src/config.py` — add `AFFECT_DECAY = 0.85`, `AFFECT_GAIN = 0.35` (interactive), keep
`AFFECT_EMA_ALPHA` for benchmark parity; document that chat uses dynamics, benchmarks use EMA.

### Tests

- `tests/test_dynamics.py` — repeated neutral input decays state toward 0; a single strong input
then neutral inputs shows monotonic decay (inertia); trajectory length == turn count.
- Update `tests/test_chat.py` — assert affect changes across consecutive turns (guards F1 regression).

### Risk / decision

- Changing default cadence alters Chat A/B reproducibility. Mitigation: benchmarks
(`benchmark_phase_chat_ab.py`) keep EMA + single-shot affect; only interactive `chat.py` uses dynamics.
Document this split in `docs/benchmarks.md`.

### Revisions from audit (AF-1, AF-3, AF-8)

- **AF-1 (FATAL):** the bug is worse than the 300s cadence — the refresh check at `engine.py:96-97`
runs **before** the user message is appended at `:99`, and an empty transcript returns
`needs_affect_refresh=False`, so **turn 1 generates on `affect_state.zero()`**. The new ordering is
strictly `append(user) → refresh(force, on trimmed msgs) → dynamics.step → _sync_affect_state → generate`, and must hold on turn 1.
- **AF-8 (STATE RISK):** do **not** stack `dynamics.step` on top of the existing `ema_update`
(`engine.py:169`) — that is double smoothing. In the chat path, `dynamics` is the **single**
integrator and writes back to `session.affect_vector`; EMA is retained only on the benchmark path.
The `lightweight=True` refresh needs a concrete backing: either implement a cheap per-turn estimator
(cached compressor + last-turn delta) or drop it and accept/measure full-TRIBEv2 per-turn latency.
- **AF-3 (METRIC LIE):** when `source` starts with `synthetic_fallback`, the synthetic field ignores
transcript content (`tribev2_client.py:15-25`), so the trajectory is not conversation-driven. The
dynamics/trajectory artifact must carry `scientific: false` and the UI must label it as synthetic.

---

## Track B: Closed perception → feeling → response loop

Split **inbound user affect** (perception) from **model internal affect** (felt state), both in the
shared 32-d space, and let the model's state be driven by perceived user affect over time.

### New files

- `src/affective/perception.py` — `estimate_user_affect(messages) -> (vec32, source)`: wraps
`**AffectEncoder`** (§1 default path), not TRIBEv2; `source` = `encoder:...` | `tribev2:...` (optional).
- `src/affective/coupling.py` — `couple(user_vec, internal_state, params) -> internal_vec`:
defines how perceived user affect drives the model's amygdala input (e.g., distress raises arousal
input current). Pure function, unit-testable, no model dependency.

### Modified files

- `src/chat/engine.py` — `refresh_affect` consumes `perception.estimate_user_affect` then
`coupling.couple` to form the SNN input current, instead of treating the transcript embedding as
both stimulus and state. Store `user_affect` and `internal_affect` separately on the session.
- `src/chat/session.py` — add `user_affect_vector`, `internal_affect_vector`; `to_log_dict` logs both.
- `src/benchmark/affect_metrics.py` — add `affect_coupling_corr(user_traj, internal_traj)`:
cross-turn correlation between user-affect delta and model-affect delta (a metric no generic
chatbot reports). Returns Pearson r + lag.

### New benchmark

- `benchmark_phase_loop.py` (Modal entrypoint) — drives a scripted multi-turn transcript with a
known affect arc (e.g., distress→hopeful) and reports coupling correlation + trajectory.
Artifact: `/artifacts/benchmarks/phase_loop.json` (+ `model_volume.commit()`).

### Tests

- `tests/test_coupling.py` — `couple` is monotonic in arousal; neutral user affect leaves internal
state on its decay path only.
- `tests/test_affect_metrics.py` (extend) — coupling corr is +1 for identical trajectories, ~0 for
random.

### Risk

- Synthetic fallback (F5) would make "coupling" an artifact of the synthetic generator, not real
perception. Mitigation: `phase_loop.json` records `source`; benchmark asserts/labels synthetic runs
as non-scientific.

### Revisions from audit (AF-3, AF-10)

- **AF-3 (FATAL for this track):** because `synthetic_fmri_timeseries(seed=42)` ignores the transcript,
coupling correlation under fallback is **meaningless** (both "user" and "internal" derive from the
same content-independent signal). `benchmark_phase_loop.py` must hard-gate `scientific: true` on
`source.startswith("tribev2:")` and refuse to report coupling for synthetic runs.
- **AF-10:** the perception→feeling closed loop is the natural home for `STDPUpdater` (reward
appropriate affect → update `amygdala` weights). Either wire it here or explicitly defer it as
exploratory; do not leave STDP orphaned.

---

## Track C: Emotion phenotype report (turn Phase-4 metrics into a profile)

Aggregate existing ablation metrics into a per-scenario "Affect Response Card." Honesty: these are
**behavioral phenotypes under modulation**, built on heuristic + log-prob metrics (F4).

### New files

- `src/benchmark/phenotype.py` — `build_phenotype(prompt_results) -> dict`: computes per-affect-state
deltas for verbosity, hedging rate, question rate, refusal/safety lexical markers, sycophancy proxy,
and reuses `affect_metrics` (logit KL, empathy/sentiment lexical, embedding cosine).
- `docs/phenotype_card_template.md` — human-readable card format for the GitHub writeup.

### Modified files

- `benchmark_phase4_extended.py` — after `prompt_ablation`, call `phenotype.build_phenotype` and add
`phenotype` block to `phase4_extended.json`.
- `src/benchmark/affect_metrics.py` — add `verbosity_delta`, `hedging_rate`, `question_rate`,
`refusal_markers` (all clearly labeled heuristic, returning relative deltas).

### Tests

- `tests/test_phenotype.py` — phenotype keys present; deltas are signed and reference the neutral
baseline; empty input yields zeros not crashes.

### Risk

- Over-claiming. Mitigation: every phenotype field carries a `"metric_type": "heuristic" | "logprob"`
tag so the card cannot present lexical scores as ground-truth emotion (auditor METRIC LIE guard).

---

## Track D: Spikes as events (spike-triggered, time-varying modulation)

Today the gate applies a static per-forward bias. Make modulation **pulse with SNN spike events** and
expose per-layer magnitude.

### Modified files

- `src/llm/hooks.py`
  - `make_hidden_state_hook`: accept an optional `gain_fn()` returning a scalar derived from current
  SNN firing (burst → larger gain). Keep additive math; gain multiplies `strength`.
  - `register_affective_hooks`: return handles **and** a small registry object exposing last-applied
  per-layer modulation norm (for the microscope UI, Track E). No behavior change when `gain_fn=None`.
- `src/chat/engine.py` — provide `gain_fn` reading `self.amygdala` last-step firing rate via
`AffectiveState` (extend state to also hold a scalar `arousal_gain`).
- `src/brain/lif_network.py` — expose last-step spike vector (already have `firing_rate_per_dim`);
add `last_spike_event` summary for gain.

### Tests

- `tests/test_hooks.py` (extend) — with `gain_fn` returning 0, output equals unmodulated; with a
constant gain, modulation scales linearly; registry records per-layer norms.

### Risk

- `patch_attention_pre_softmax` already disables cache (auditor 2.5). Do **not** route event gain
through that path for chat; keep it on the residual hook to preserve long-context KV. Documented.

### Revisions from audit (AF-5, AF-9, AF-11)

- **AF-5 (INVARIANT BREAK):** changing only `LIFAmygdala.forward` does **not** deliver carryover,
because chat runs the SNN via `extract_signature_from_pipeline` (`signatures.py:115-116`) and
`sequence_affective_vectors` (`lif_network.py:93-113`), both of which call `init_leaky()` per call.
Add `src/chat/signatures.py` to this change and thread `snn_mem_state` through both helpers. The
`return_state=False` 2-tuple default is confirmed sufficient to protect the 8 existing `forward()`
callers.
- **AF-9 (EDGE CASE):** `strength` only affects additive mode; scale mode multiplies regardless
(`hooks.py:58-61`), and the hook treats a zero *tensor* as active (`hooks.py:49`). Assert **additive
mode for chat**, and define neutral as **hooks-off**, so a trained gate bias cannot leak through a
zeroed affect vector.
- **AF-11 (EDGE CASE):** removing `init_leaky()` lets membrane drift unboundedly over a long session.
Define a `reset_state()` cadence (on `/refresh`, profile switch, session clear) and add a
bounded-membrane stability test over many turns.

---

## Track E: The Emotion Microscope (the flagship demo)

A live, split-view interface showing the mechanism moving in sync with language. This is the
"talk about it on GitHub" centerpiece.

### New files

- `src/serve/microscope_api.py` — FastAPI app exposing `/chat` (returns reply + affect vector +
per-layer hook norms + firing stats + rolling logit-KL vs neutral + `source`) and `/state`.
Wraps `ChatEngine`; runs locally or behind the Modal worker.
- `web/microscope/` — minimal React/Vite app: left pane (hooks off / neutral), right pane (full
affective pipeline), live panel (32-d bar chart, firing-rate sparkline, per-layer hook magnitude,
rolling KL). Paste-a-transcript button to drive an affect arc.
- `run_microscope.py` — Modal/ASGI entrypoint serving `microscope_api` + static web build.

### Modified files

- `src/chat/engine.py` — add `generate_reply(..., return_introspection=True)` that also returns
per-layer hook norms (from Track D registry), firing stats, and current vector. Must **not** change
default return contract (benchmarks depend on it).
- `src/benchmark/affect_metrics.py` — reuse `kl_divergence_from_logits` to compute the live rolling KL
(neutral logits vs modulated logits at the latest token); document it is last-token KL (auditor 5.2).
- `requirements.txt` — add `fastapi`, `uvicorn`; `requirements-vllm.txt` untouched.

### Tests

- `tests/test_microscope_api.py` — `/chat` returns all introspection keys and a valid `source`;
neutral pane KL ≈ 0; smoke test with synthetic fallback labeled as such.

### Risk

- Two model passes (neutral + modulated) per token doubles cost. Mitigation: compute neutral logits
only at the last prompt token (not per generated token) for the rolling KL, matching the existing
metric's validity window.

### Revisions from audit (AF-3, AF-6, AF-7)

- **AF-7 (METRIC LIE):** `last_token_logits` calls `model(**inputs)` which **fires the persistent chat
hooks** (`affect_metrics.py:90,95`; hooks from `engine.py:57`), so a naive "neutral" pass returns
*modulated* logits. The neutral pass must be a **hooks-off pass on the same model instance** (remove
handles or force additive `strength=0`). Rename the metric "last-prompt-token KL (updated per turn)"
— it is not per-generated-token.
- **AF-6 (STATE RISK):** `microscope_api` must **not** share one `ChatEngine`/`affect_state`/hook set
across concurrent requests (same bug as the Modal worker, `run_chat.py:31`) — `_sync_affect_state`
from one request would corrupt another's in-flight hooks. Use a per-session engine or a generation
lock; add a concurrency test.
- **AF-3 (METRIC LIE):** the live panel must display `source`; under `synthetic_fallback` it must show
a clear "synthetic — not conversation-driven" badge rather than implying real perceived affect.

---

## Track F: Brain-aligned validation (v2 optional — see §1.6)

Tie the **encoder's 32-d space** to measured ROI patterns (OpenNeuro Alice), not TRIBEv2 predictions.

### New files

- `src/benchmark/brain_alignment.py` — correlate encoder VAD dimensions with ROI-derived valence/arousal
on held-out Alice segments; report r, CI, and `scientific: true` only when p < 0.05 on held-out data.
- `benchmark_phase_brain.py` — Modal entrypoint; artifact `/artifacts/benchmarks/phase_brain.json`.

### Modified files

- `docs/benchmarks.md` — add "Phase brain — encoder vs measured ROI (v2)" with metric defs; negative
results are valid publishable outcomes.

### Tests

- `tests/test_brain_alignment.py` — alignment metric bounded; runs without ds322 preprocessing flagged
`scientific: false`.

### Risk

- Amygdala ROI correlation may be weak or null at 3T — publish honestly. Primary alignment targets are
insula/ACC/temporal parcels (§1.6). TRIBEv2-subcortical may appear as an optional comparison channel,
never as ground truth.

---

## Track G: Personalization / affect profiles (product path)

Different SNN initial conditions + hook maps = distinct "temperaments," plus light user calibration.

### New files

- `src/chat/profiles.py` — named profiles (`calm_therapist`, `enthusiastic_coach`, `neutral_analyst`):
initial SNN membrane bias, `AFFECT_DECAY`/`AFFECT_GAIN`, default `hook_strength`, target layers.
- `src/chat/calibration.py` — 3-turn onboarding mapping user feedback ("too cold/too much/just right")
to strength + decay adjustments.

### Modified files

- `chat.py` — add `--profile` flag and `/profile NAME` command; `_handle_command` wiring.
- `run_chat.py` — `EmotionalChatWorker.set_profile` Modal method.
- `src/chat/session.py` — persist `profile_name` and calibration offsets in `to_log_dict`.
- `src/config.py` — `DEFAULT_PROFILE = "neutral_analyst"`, `PROFILES` registry table.

### Tests

- `tests/test_profiles.py` — each profile yields distinct initial state; calibration moves params in
the expected direction and clamps to safe ranges.

### Risk / ethics

- Longitudinal user-affect memory raises privacy concerns. Plan keeps it **opt-in, local, encrypted,
off by default**; documented in the GitHub writeup. No remote storage in this track.

### Revisions from audit (AF-6)

- Profiles/calibration mutate engine state. In the multi-user Modal worker, `set_profile` on a shared
engine would change another caller's temperament mid-generation. Bind profile + calibration to a
per-session engine (same fix as AF-6 in Track E); never mutate a globally shared engine.

---

## Track H: Multimodal affect (later — same amygdala pathway)

Reuse the 32-d space for voice prosody and (optionally) facial expression so it's "process emotion,"
not "classify sentiment."

### New files

- `src/encoder/prosody.py` — F0/jitter/pause features → delta-mod spikes into the same SNN input
(`delta_modulate`, identical theta).
- `src/encoder/vision_adapter.py` — (optional) facial-expression features → 32-d via a small adapter.

### Modified files

- `src/affective/tribev2_client.py` — `run_tribev2_predict` already accepts `video_path`; add a
documented audio path and ensure `source` distinguishes modality (`tribev2:video`, `prosody:...`).
- `src/config.py` — feature dims + `DELTA_THETA` reuse note (must match text path; auditor invariant 1).

### Tests

- `tests/test_prosody.py` — prosody features map to 32-d with correct shape and theta; spike sparsity
in expected band.

### Risk

- Adding modalities can silently change `DELTA_THETA` behavior. Mitigation: one theta, asserted equal
across encoders.

---

## Sequencing & Milestones


| Milestone                    | Tracks                  | Why this order                                                                      |
| ---------------------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| **M0 — "Data that teaches"** | §1                      | EmpatheticDialogues + encoder + holdout registry — prerequisite for honest training |
| M1 — "Real, not random"      | I + §1                  | Modulation trained on labeled dialogue, not TRIBEv2 (F2, AF-2)                      |
| M2 — "Emotion over time"     | A, D                    | Per-turn dynamics + spike-triggered gain + membrane carryover (F1/F3)               |
| M3 — "See it move"           | E                       | Microscope demo — highest GitHub impact, depends on A/D introspection               |
| M4 — "Closed loop"           | B                       | Perception→feeling coupling + coupling-corr metric                                  |
| M5 — "Credibility"           | C, F (v2 fMRI optional) | Phenotype cards + optional Alice alignment                                          |
| M6 — "Product/scale"         | G, H                    | Profiles, calibration, multimodal                                                   |


**Minimum viable wow:** M0 + M1 + M2 + M3.

---

## Config Changes Summary (`src/config.py`)

- `EMPATHETICDIALOGUES_DIR`, `SCENARIO_HOLDOUT_DIR`, `AFFECT_ENCODER_CKPT` (§1)
- `GATE_CKPT_NAME`, `AMYGDALA_CKPT_NAME`, `AFFECT_HIDDEN_SIZE` (Track I)
- `AFFECT_DECAY = 0.85`, `AFFECT_GAIN = 0.35` (Track A; chat-only, EMA retained for benchmarks)
- `DEFAULT_PROFILE`, `PROFILES` (Track G)
- Prosody/vision feature dims, shared `DELTA_THETA` assertion (Track H)

## New Tests Summary

`test_emotion_lexicon.py`, `test_affect_encoder.py`, `test_empatheticdialogues_dataset.py`,
`test_checkpoints.py`, `test_gate_training.py`, `test_dynamics.py`, `test_coupling.py`,
`test_phenotype.py`, `test_microscope_api.py`, `test_brain_alignment.py`, `test_profiles.py`,
`test_prosody.py`, plus extensions to `test_chat.py`, `test_hooks.py`, `test_affect_metrics.py`.

## New Modal Entrypoints / Artifacts

- `train_affect_encoder.py` → `/models/affect/train_encoder.json`
- `scripts/download_empatheticdialogues.py` (local data prep)
- `benchmark_phase_loop.py` → `phase_loop.json`
- `benchmark_phase_brain.py` → `phase_brain.json`
- `run_microscope.py` (serving) — every artifact write paired with `model_volume.commit()`.

---

## Cross-Cutting Invariants Checklist (apply to every PR)

- [ ] `AFFECT_DIM == 32` preserved through new code paths.
- [ ] Training uses §1 labeled corpus; TRIBEv2 never used as loss target in v1.
- [ ] `source` surfaced in artifact + UI (`encoder:...`, `tribev2:...`, `synthetic_fallback:...`).
- [ ] Synthetic / TRIBEv2-demo runs never labeled scientific; holdout prompts never in train split.
- [ ] Affect computed on the **same** transcript text generation sees (F6).
- [ ] Refresh → sync → generate ordering on every turn.
- [ ] Hooks removed in `finally`/`cleanup`; no leaked handles in long-lived workers.
- [ ] Heuristic metrics tagged `metric_type`; never presented as ground-truth emotion.
- [ ] Benchmark A/B differs only by affect vector (prompt/temp/tokens/strength/seed fixed).
- [ ] `model_volume.commit()` after every Modal artifact write.
- [ ] Weight-quant (NF4) and KV-quant (INT8/INT4/FP8) never conflated in new docs/code.

---

## Changelog of Planned Changes (living log)

> Updated as the audit refines the plan. "Status" = planned / revised / rejected.


| ID      | Change                                                        | Files                                                                                                                                                       | Status                                                             |
| ------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| C-DATA1 | Post-TRIBEv2 supervision: EmpatheticDialogues + encoder stack | `emotion_lexicon.py`, `encoder.py`, `dataset.py`, `pipeline.py`, `perception.py`, `train_affect_encoder.py`, `download_empatheticdialogues.py`, `config.py` | **M0 implemented** (P0 audit fixes applied; not wired to chat yet) |
| C-I1    | Train + load gate/amygdala checkpoints                        | `train_gate.py`, `checkpoints.py`, `engine.py`, `train_snn.py`, `hybrid_runner.py`, `config.py`                                                             | revised (AF-2, AF-4, AF-10; depends on C-DATA1)                    |
| C-A1    | Per-turn affect dynamics + membrane carryover                 | `dynamics.py`, `lif_network.py`, `engine.py`, `session.py`, `config.py`                                                                                     | revised (AF-1, AF-3, AF-8)                                         |
| C-B1    | Perception/feeling split + coupling metric                    | `perception.py`, `coupling.py`, `engine.py`, `affect_metrics.py`, `benchmark_phase_loop.py`                                                                 | revised (AF-3, AF-10)                                              |
| C-C1    | Emotion phenotype card                                        | `phenotype.py`, `benchmark_phase4_extended.py`, `affect_metrics.py`                                                                                         | planned                                                            |
| C-D1    | Spike-triggered modulation + per-layer norms                  | `hooks.py`, `engine.py`, `lif_network.py`, `signatures.py`                                                                                                  | revised (AF-5, AF-9, AF-11)                                        |
| C-E1    | Emotion Microscope API + web UI                               | `microscope_api.py`, `web/microscope/`, `run_microscope.py`, `engine.py`                                                                                    | revised (AF-3, AF-6, AF-7)                                         |
| C-F1    | Brain-alignment benchmark                                     | `brain_alignment.py`, `benchmark_phase_brain.py`, `benchmarks.md`                                                                                           | planned                                                            |
| C-G1    | Profiles + calibration                                        | `profiles.py`, `calibration.py`, `chat.py`, `run_chat.py`, `session.py`, `config.py`                                                                        | revised (AF-6)                                                     |
| C-H1    | Multimodal encoders                                           | `prosody.py`, `vision_adapter.py`, `tribev2_client.py`, `config.py`                                                                                         | planned                                                            |


### Audit Findings Log (business-logic-auditor review)

> All six current-state findings (F1–F6) were **confirmed against source**; F1, F5, F6 were
> *understated*. Eleven findings below, three FATAL-class. Plan tracks have been revised accordingly
> (see "Revisions from audit" notes inside each affected track).

#### AF-1 — Affect lags the turn; turn-1 runs on zeroed affect — [FATAL LOGIC] → C-A1 / §0 F1

`engine.py:96-97` runs the refresh check **before** `engine.py:99` appends the user message, and
`needs_affect_refresh` returns `False` on an empty transcript (`session.py:43-46`), so the opening
reply generates with `affect_state.zero()` (`engine.py:52-53`). Affect always reflects the transcript
*minus* the latest user turn. **Edit:** adopt `append → refresh → dynamics.step → sync → generate`,
covering turn 1.

#### AF-2 — Track I would load an *untrained* SNN and call it "trained" — [FATAL LOGIC] → C-I1

`train_snn.py:43` builds a random `LIFAmygdala` and `:75` saves it with **no affect loss/labels/grad
step**; only optional STDP touches `fc1`, clamped to [0,1] (`stdp.py:22-23`), never `lif`/`fc2`.
Loading `amygdala.pt` yields ~random modulation. **Edit:** add a real supervised/contrastive amygdala
objective, **or** relabel `amygdala.pt` as "frozen untrained brain-derived geometry" and drop the
"trained / AI feels" framing for the SNN; fix/document the STDP clamp range.

#### AF-3 — Synthetic fallback is content-independent; per-turn dynamics are theater — [METRIC LIE] → C-A1, C-B1, C-E1 / §0 F5

`run_tribev2_predict` swallows all errors → synthetic fMRI (`tribev2_client.py:105-107`), and
`synthetic_fmri_timeseries(seed=42)` **ignores the transcript entirely** (`:15-25`). Under fallback
every refresh returns the same field → trajectories/coupling/microscope decoupled from user content.
**Edit:** fail closed when `source` starts with `synthetic_fallback` — suppress trajectory/coupling
claims, mark run non-scientific in artifact + UI.

#### AF-4 — Gate training breaks the implicit `gate(0)=0` neutral baseline — [INVARIANT BREAK] → C-I1

Neutral is currently the zero affect vector, relying on `AffectiveGate` zero-init bias
(`hooks.py:18`; `hybrid_runner.py:17`). A trained gate no longer guarantees `gate(0)=0`, so every
"differs only in affect vector" comparison silently runs against a nonzero-modulated baseline; a soft
regularizer is not exact. **Edit:** redefine neutral as a **hooks-off** pass for `hybrid_runner`/Phase-4
and add a hard no-op assertion (`‖gate(zeros)‖ < ε`) at checkpoint save/load.

#### AF-5 — Track D fixes `forward` but the chat path still resets membrane — [INVARIANT BREAK] → C-D1 / §0 F3

The chat path never calls `amygdala.forward` directly — it goes through
`extract_signature_from_pipeline` (`engine.py:162`) → `sequence_affective_vectors`
(`lif_network.py:93-113`) + `amygdala(...)` (`signatures.py:115-116`), both calling `init_leaky()`
per invocation. Persisting the instance ≠ persisting membrane. (Backward-compat `return_state=False`
2-tuple default **is** sufficient for all 8 existing `model()` callers — verified.) **Edit:** add
`extract_signature_from_pipeline` + `sequence_affective_vectors` to C-D1 and thread `snn_mem_state`.

#### AF-6 — Shared engine/session/hooks → concurrent cross-contamination — [STATE RISK] → C-E1, C-G1

`EmotionalChatWorker` holds one `self.engine`/`self.session`/`self.affect_state` and one hook closure
(`run_chat.py:31`); Track E's `microscope_api` wraps a single `ChatEngine`. Concurrent calls interleave:
one request's `_sync_affect_state` overwrites the shared `affect_state` another's in-flight hooks read
(`hooks.py:48`), and both append to `session.messages`. **Edit:** per-request `ChatEngine`/`ChatSession`
(or a generation lock) + concurrency test; never share `affect_state`/hooks across simultaneous gens.

#### AF-7 — "Rolling KL" neutral pass fires live hooks and is mislabeled — [METRIC LIE] → C-E1

`last_token_logits` calls `model(**inputs)` (`affect_metrics.py:90,95`), which triggers the persistent
chat hooks (`engine.py:57`), so a naive neutral pass returns *modulated* logits; KL is computed only at
the last prompt token. **Edit:** neutral logits via a **hooks-off pass on the same instance** (remove
handles or additive `strength=0`); rename to "last-prompt-token KL (updated per turn)."

#### AF-8 — Double integration (EMA + dynamics) and an unbacked `lightweight` path — [STATE RISK] → C-A1

`refresh_affect` already applies `ema_update` (`engine.py:169`); layering `dynamics.step` composes two
smoothers. The proposed `lightweight=True`/"incremental transcript" has no backing API —
`run_tribev2_predict` only accepts `text_path`/`video_path` (`tribev2_client.py:76-104`). **Edit:** use
a single integrator in the chat path (drop EMA when dynamics active), define dynamics↔`affect_vector`
write-back, and either implement a concrete cheap estimator for `lightweight` or document full-TRIBEv2
per-turn latency.

#### AF-9 — `strength=0` doesn't disable scale mode; zeroed affect still injects bias — [EDGE CASE] → C-D1

`strength` is applied only in additive mode; scale mode does `hidden = hidden * mod` regardless
(`hooks.py:58-61`). The hook skips only `None` (`hooks.py:49`), so a zero *tensor* is "active" and
still injects `gate` bias once that bias is nonzero (post-training). **Edit:** assert additive mode for
chat (or make scale honor `strength`); define neutral as hooks-off rather than zeroed-affect.

#### AF-10 — STDP module is orphaned from the plan's learning story — [STATE RISK] → C-I1, C-B1

`STDPUpdater` is exercised only by `train_snn.py --stdp-steps` and clamps to [0,1]; Track I training and
Track B closed-loop feedback never use it. **Edit:** wire STDP into the perception→feeling closed loop
(its natural home) **or** mark it exploratory in `docs/benchmarks.md`; add `tests/test_stdp.py`.

#### AF-11 — Unbounded membrane drift across turns, no reset cadence — [EDGE CASE] → C-D1 / §0 F3

Once `init_leaky()` is removed for carryover, membrane state can accumulate without bound over an
open-ended session; the plan mentions `reset_state()` but no trigger policy. **Edit:** define a reset
cadence (on `/refresh`, profile switch, session clear) + a bounded-drift numerical-stability test.


| Finding | Severity        | Plan section             | Required plan change                                                                                |
| ------- | --------------- | ------------------------ | --------------------------------------------------------------------------------------------------- |
| AF-1    | FATAL LOGIC     | C-A1 / §0 F1             | Document pre-append ordering + turn-1 zeroed affect; reorder append→refresh→dynamics→sync→generate. |
| AF-2    | FATAL LOGIC     | C-I1                     | Random SNN saved as trained; add real amygdala objective or relabel as untrained geometry.          |
| AF-3    | METRIC LIE      | C-A1, C-B1, C-E1 / §0 F5 | Synthetic fMRI ignores transcript; fail closed, mark synthetic non-scientific, suppress claims.     |
| AF-4    | INVARIANT BREAK | C-I1                     | Trained gate breaks `gate(0)=0`; neutral = hooks-off pass + hard no-op assertion.                   |
| AF-5    | INVARIANT BREAK | C-D1 / §0 F3             | Thread `mem` through `extract_signature_from_pipeline` + `sequence_affective_vectors`.              |
| AF-6    | STATE RISK      | C-E1, C-G1               | Per-request engine/session or lock; concurrency test.                                               |
| AF-7    | METRIC LIE      | C-E1                     | Neutral logits via hooks-off pass on same instance; rename to last-prompt-token KL.                 |
| AF-8    | STATE RISK      | C-A1                     | Single integrator (drop EMA under dynamics); back the `lightweight` path.                           |
| AF-9    | EDGE CASE       | C-D1                     | `strength=0` only disables additive; assert additive for chat; zeroed affect leaks bias.            |
| AF-10   | STATE RISK      | C-I1, C-B1               | Wire STDP into closed loop or mark exploratory; add `tests/test_stdp.py`.                           |
| AF-11   | EDGE CASE       | C-D1 / §0 F3             | Define membrane reset cadence + bounded-drift stability test.                                       |


