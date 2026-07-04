# Verification Session Log — Post-M1 Solidification

> **Date:** 2026-06-22  
> **Scope:** Wire M1 checkpoints into benchmarks, add Track B/C/F verification tooling, run tests + Modal behavioral benchmarks, code audit.  
> **Prior training results:** [results.md](./results.md)  
> **Implementation plan:** [teaching_ai_emotion_plan.md](./teaching_ai_emotion_plan.md)

---

## 1. What was implemented this session


| Change                                                                                       | Milestone / track | Purpose                                                  |
| -------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| `hybrid_runner.py` loads trained encoder, amygdala, gate                                     | M1 / I            | Benchmarks use real checkpoints, not ran;dom weights     |
| `benchmark_phase4_extended.py` → `affective_image`, neutral = **hooks-off**, phenotype block | M5 / C            | Behavioral validation with honest neutral baseline       |
| `benchmark_phase_chat_ab.py` → encoder path (not TRIBEv2)                                    | M4 / B            | Transcript-conditioned affect from trained stack         |
| `benchmark_phase_loop.py` (new)                                                              | M4 / B            | Multi-turn coupling correlation on distress/hopeful arcs |
| `src/benchmark/phenotype.py`                                                                 | M5 / C            | Heuristic phenotype card from ablation rows              |
| `src/benchmark/brain_alignment.py`                                                           | M5 / F            | Stub alignment report (`scientific: false` without fMRI) |
| `scripts/run_m1_verification.py`                                                             | —                 | Reproducible local verification harness                  |


**Not implemented (deferred):** M6 profiles/multimodal, full Emotion Microscope React UI.

**Implemented (2026-06-20):** Gate v2 training objective, M2 SNN membrane carryover (`snn_mem_state` in session/engine), M3 minimal Microscope API (`src/serve/microscope_api.py`).

---

## 2. Step-by-step verification log

### Step 1 — Full pytest suite

**Command:**

```powershell
py -3 -m pytest tests/ -q --ignore=tests/test_hybrid_encoder.py
```

**Result:** **81 passed, 4 skipped, 0 failed**


| Skipped test file        | Reason                                  |
| ------------------------ | --------------------------------------- |
| `test_microscope_api.py` | Track E API (mocked engine) |
| `test_profiles.py`       | Track G not built                       |
| `test_prosody.py`        | Track H not built                       |
| `test_gate_training.py`  | Requires CUDA for full gate train smoke |


**Interpretation:** Unit/integration coverage for M0/M1, dynamics, coupling, hooks, checkpoints, phenotype, and brain-alignment stub is green. Skips are expected placeholders.

---

### Step 2 — Local verification harness

**Command:**

```powershell
py -3 scripts/run_m1_verification.py
```

**Artifact:** `data/artifacts/verification_report.json`


| Sub-step                  | Result                                                    | Interpretation                                                                                                      |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `pytest_full_suite`       | exit 0                                                    | Same as Step 1                                                                                                      |
| `checkpoint_paths`        | Local dirs exist under `data/artifacts/{affect,snn,gate}` | Modal volume holds canonical trained weights; local dirs may be empty or stale — benchmarks on Modal use `/models/` |
| `encoder_fixture_vad_mae` | mean **0.357** on 1 fixture sample (hash backend)         | Hash encoder is weak by design; not comparable to hybrid Modal `test_vad_mae ≈ 0.16`                                |
| `coupling_distress_arc`   | coupling_corr **1.0** on synthetic emotion trajectory     | Metric works on aligned synthetic deltas; not a scientific claim                                                    |
| `phenotype_builder`       | empathy_delta **0.91** on toy strings                     | Builder correctly ranks empathetic vs neutral text                                                                  |
| `brain_alignment_stub`    | `scientific: false`, `r: 0.0`                             | Correctly refuses fMRI claims without data                                                                          |
| `gate_noop_random_init`   | pass                                                      | Fresh gate satisfies AF-4 at init                                                                                   |


**Overall:** 7/7 local steps passed.

---

### Step 3 — Modal `benchmark_phase_loop` (Track B)

**Command:**

```powershell
modal run benchmark_phase_loop.py
```

**Modal app:** [ap-RJkqDeqgraPqrnrStC6PNA](https://modal.com/apps/hrudayiitb/main/ap-RJkqDeqgraPqrnrStC6PNA)  
**Artifact:** `/models/benchmarks/phase_loop.json`


| Arc      | Turns | Source                           | Encoder | Amygdala | coupling_corr |
| -------- | ----- | -------------------------------- | ------- | -------- | ------------- |
| distress | 3     | `encoder:empatheticdialogues_v1` | trained | trained  | **-1.0**      |
| hopeful  | 3     | `encoder:empatheticdialogues_v1` | trained | trained  | **-1.0**      |


`scientific: true` (encoder-driven, not synthetic fallback).

**Interpretation:**

- Checkpoints **did load** on Modal (`encoder_source` / `amygdala_source`: trained).
- **coupling_corr = −1.0** is suspicious for a "user/internal co-movement" claim: with only **2 consecutive deltas** from 3 turns, anti-correlated magnitude changes produce ±1. Pearson r is fragile at this length. Do **not** publish this as evidence of coupling; treat as a **metric calibration TODO** (longer arcs, more turns, sign-aware coupling).
- `scientific: true` here means "not synthetic TRIBEv2 fallback" — not "statistically significant."

---

### Step 4 — Modal `benchmark_phase4_extended` (behavioral)

**Command:**

```powershell
modal run benchmark_phase4_extended.py --skip-strength-sweep
```

**Modal app:** [ap-4Cx9ybTtlI5fsTephPT517](https://modal.com/apps/hrudayiitb/main/ap-4Cx9ybTtlI5fsTephPT517)  
**Artifact:** `/models/benchmarks/phase4_extended.json`


| Summary metric                                  | Value           |
| ----------------------------------------------- | --------------- |
| Prompts                                         | 5               |
| Text changed (hooks-off neutral vs high affect) | **4 / 5 (80%)** |
| Mean logit KL (neutral → high)                  | **0.0085**      |
| Mean embedding cosine distance                  | **0.049**       |


**Phenotype card (heuristic):**


| Field           | Delta      |
| --------------- | ---------- |
| empathy_delta   | **−0.008** |
| sentiment_delta | 0.0        |
| verbosity_delta | −2.4       |
| hedging_delta   | +0.0057    |


**Interpretation:**

- **Affect modulates generation** in 80% of holdout prompts (fixed temp=0, strength=1) — supports the core "neuromodulation changes outputs" claim at a coarse level.
- Logit KL ~0.0085 is **small** — affect shifts last-token distribution slightly, not radically (expected for additive bias on 1B model).
- **empathy_delta ≈ 0** (slightly negative) means high affect did **not** reliably increase lexical empathy markers vs hooks-off neutral on these 5 prompts. Gate training (100 samples, 1 epoch) is **not yet sufficient** for empathy phenotype claims.
- Phenotype fields are tagged `metric_type: heuristic` — suitable for exploratory cards, not publication.

**Neutral baseline fix (AF-4):** ablation `neutral` condition now uses `hooks_enabled=False`, not zero-vector + hooks on.

---

### Step 5 — Code audit (business-logic-auditor)

**Scope:** Uncommitted changes vs AF-1–AF-11 in the implementation plan.

#### Confirmed improvements


| Finding                                     | Status after this session                              |
| ------------------------------------------- | ------------------------------------------------------ |
| AF-2 untrained SNN/gate in benchmarks       | **Mitigated** — `hybrid_runner` loads checkpoints      |
| AF-4 neutral = hooks-off in Phase 4         | **Fixed** in ablation + KL `hooks_off_a`               |
| AF-3 synthetic TRIBEv2 in chat A/B script   | **Mitigated** — `benchmark_phase_chat_ab` uses encoder |
| Metric honesty (phenotype, brain_alignment) | **Improved** — `metric_type` / `scientific: false`     |


#### Remaining risks (not fixed this session)


| ID        | Severity            | Issue                                                                                                                                                                       | Location                                     |
| --------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| AF-1 / F6 | **STATE RISK**      | `ChatEngine` registers hooks at init and keeps them on for all generations; neutral chat turns still run **with hooks active** unless caller uses a separate hooks-off path | `src/chat/engine.py:75-85`, `128-135`        |
| AF-3      | **METRIC LIE risk** | `phase_loop` can label `scientific: true` while coupling_corr is meaningless at 3 turns                                                                                     | `benchmark_phase_loop.py`                    |
| AF-5 / F3 | **STATE RISK**      | SNN `init_leaky()` each forward — no membrane carryover across turns                                                                                                        | `src/brain/lif_network.py`, `signatures.py`  |
| AF-7      | **METRIC LIE**      | `last_token_logits` during hooked generation still used in some paths without hooks-off neutral pass                                                                        | audit existing `affect_metrics.py` consumers |
| AF-6      | **STATE RISK**      | Shared `ChatEngine` across Modal workers if not per-session                                                                                                                 | `run_chat.py`                                |
| AF-9      | **EDGE**            | Scale mode on gate bypasses strength; chat uses additive only — OK if documented                                                                                            | `hooks.py`                                   |
| M3/E      | **Deferred**        | No microscope API — cannot demo live affect                                                                                                                                 | planned                                      |


**Audit conclusion:** M1 training + benchmark wiring are **honest and test-backed** for "modulation changes outputs." Claims about **empathy quality**, **temporal coupling**, and **neutral chat behavior** need the remaining AF fixes and longer gate training before public GitHub narrative.

#### Audit remediations applied (post [Business logic code audit](7fa938e8-4cd9-4347-a13c-91790c912e44))


| ID  | Fix                                                                                            |
| --- | ---------------------------------------------------------------------------------------------- |
| N-1 | Strength-sweep KL uses `hooks_off_a=True`                                                      |
| N-2 | Phase loop threads `couple(u, state)` + `dyn.step`                                             |
| N-3 | `scientific` requires `encoder_source` and `amygdala_source` == `trained`                      |
| N-4 | `build_affect_vectors` raises on zero SNN output (no silent 0.25 fallback)                     |
| N-5 | Phenotype prefers `generated_text` over 400-char preview                                       |
| N-6 | `load_amygdala` / `load_encoder` return `unverified_checkpoint` without `supervision` metadata |


---

## 3. Milestone status after verification


| Milestone                | Status      | Evidence                                                  |
| ------------------------ | ----------- | --------------------------------------------------------- |
| **M0** Data + encoder    | Done        | Training + tests                                          |
| **M1** Real checkpoints  | Done        | Modal `/models/{affect,snn,gate}` + checkpoint tests      |
| **M2** Emotion over time | **Done** | `AffectDynamics` + SNN `snn_mem_state` carryover in chat |
| **M3** Microscope        | **Partial** | FastAPI `/chat`, `/state`, `/reset`; engine introspection |
| **M4** Closed loop       | **Partial** | `coupling.py`, `phase_loop` benchmark (metric needs work) |
| **M5** Credibility       | **Partial** | Phase 4 + phenotype; brain alignment stub only            |
| **M6** Product           | Not started | —                                                         |


---

## 4. Recommended next steps (priority order)

1. ~~**Chat engine AF-1 / AF-4:** Remove persistent hooks; register only when `affect_vector` is non-neutral and strength > 0; hooks-off path for neutral generation.~~ **Done** (2026-06-22).
2. ~~**Re-run gate training** with more samples/epochs; re-run Phase 4; target **positive empathy_delta** on emotional_support prompt.~~ **Done** — 500 samples × 3 epochs; Phase 4 empathy_delta **+0.037**; transcript-conditioned chat/scenario still collapses (see [results.md](./results.md#behavioral-verification-post-gate-re-train)).
3. ~~**Fix coupling benchmark:** longer scripted arcs (≥8 turns), report CI or label `exploratory: true` when n_deltas < 5.~~ **Done**
4. ~~**Run `modal run benchmark_phase_chat_ab.py`** with new encoder path.~~ **Done** — collapse on distress/hopeful; neutral OK.
5. ~~**Run `modal run benchmark_phase_scenarios.py`** on holdout scripts.~~ **Done** — `scientific: true` but hooks-on repetition.
6. ~~**Fix gate objective** — repetition penalty / sequence loss; re-run scenario + chat A/B before quality claims.~~ **Gate v2 code done** — Modal re-verify pending.
7. ~~**M2:** Thread SNN membrane state through `extract_signature_from_pipeline` / `sequence_affective_vectors`.~~ **Done**
8. ~~**M3:** Minimal `microscope_api.py` + introspection fields on `generate_reply`.~~ **Done**

---

## 5. Commands reference

```powershell
# Local
py -3 scripts/run_m1_verification.py
py -3 -m pytest tests/ -q --ignore=tests/test_hybrid_encoder.py

# Modal behavioral (use py -3 -m modal if modal not on PATH)
py -3 -m modal run train_gate.py --max-samples 500 --epochs 3
py -3 -m modal run benchmark_phase_loop.py
py -3 -m modal run benchmark_phase4_extended.py --skip-strength-sweep
py -3 -m modal run benchmark_phase_chat_ab.py --max-new-tokens 64
py -3 -m modal run benchmark_phase_scenarios.py --max-new-tokens 64
py -3 scripts/run_behavioral_verification.py
py -3 run_microscope.py
py -3 scripts/generate_scenarios.py
py -3 scripts/run_scenario_eval_local.py
```

---

## 6. Artifact index


| Path                                      | Contents                               |
| ----------------------------------------- | -------------------------------------- |
| `data/artifacts/verification_report.json` | Local harness JSON                     |
| `data/artifacts/modal_phase_loop.log`     | Phase loop stdout                      |
| `data/artifacts/modal_phase4.log`         | Phase 4 stdout                         |
| `/models/benchmarks/phase_loop.json`      | Coupling arc results (Modal volume)    |
| `/models/benchmarks/phase4_extended.json` | Ablation + phenotype (Modal volume)    |
| `docs/results.md`                         | M1 training metrics and interpretation |


