# Interactive chat hardening plan

Gate v3.1 is **collapse-free in benchmarks**, but the live chat path (`chat.py`, `run_chat.py`, `ChatEngine`, Microscope API) has not yet been held to the same standard. This plan closes that gap.

**Goal:** A user can run multi-turn emotional chat (local CUDA or Modal worker) with confidence that (1) generation will not collapse into empathy-token loops, (2) affect modulation is observable and controllable, and (3) failures are detected and surfaced instead of silently degrading.

---

## Status (updated 2026-07-04)

| Phase | Status |
|-------|--------|
| 1A — collapse guard in `ChatEngine.generate_reply` | ✅ Implemented + unit-tested (`tests/test_chat_engine_collapse_guard.py`) |
| 1B — gate load/version verification | ✅ Implemented (`ChatEngine.gate_health()`, Modal worker fail-fast in `setup()`) + unit-tested |
| 2C — per-turn session metrics, schema v2 | ✅ Implemented (`ChatSession.turn_metrics`, `to_log_dict()["chat_log_schema"] == 2`) + unit-tested |
| 2D — CLI collapse banner + `/status` | ✅ Implemented in `chat.py` / `src/chat/tone_markers.py` |
| 3E — chat soak benchmark | ✅ Script written (`benchmark_phase_chat_soak.py`) — **not yet executed on Modal GPU**; no empirical 256-token/10-turn result yet |
| 3F — no-GPU regression tests for soak | ✅ `tests/test_chat_soak_regression.py` |
| 4 — Modal worker production hygiene | ✅ `setup()` fail-fasts on untrained gate, warns on version mismatch; `get_health()` method added |
| 5 — Microscope API alignment | ✅ `/chat` returns collapse/gate fields; `/health/{session_id}` endpoint added + unit-tested |

**Remaining before full sign-off:** run `py -3 -m modal run benchmark_phase_chat_soak.py` on GPU and confirm `summary.passed == true`, then update `data/README.md` "What we cannot claim yet" with the empirical result.

---

## Current state

| Component | Status |
|-----------|--------|
| `chat.py` CLI | Works: local CUDA or Modal `EmotionalChatWorker`; `/mood`, `/refresh`, `/strength`, `/affect`, `/save` |
| `ChatEngine.generate_reply` | Correctly decodes **new tokens only** for replies; registers hooks per generation (AF-4) |
| `run_chat.py` Modal worker | Warm GPU class; loads gate from volume; no collapse guard |
| `run_microscope.py` API | FastAPI wrapper with introspection; shares `ChatEngine` |
| Benchmark parity | Chat A/B uses same encoder→SNN→gate→hooks path; collapse detector fixed 2026-07-04 |
| Tests | `tests/test_chat.py`, `tests/test_m1_engine.py`, `tests/test_microscope_api.py` — unit-level only |

**Gaps (original; see Status above for current state):**

1. ~~No runtime `detect_empathy_collapse` on chat replies (256-token generations).~~ Fixed — Phase 1A.
2. ~~No automatic backoff (lower strength / disable hooks) when collapse is detected mid-session.~~ Fixed — Phase 1A (hooks-off retry).
3. ~~No structured session log with per-turn `new_text`, collapse score, hook strength, affect norm.~~ Fixed — Phase 2C.
4. ~~Modal worker has no health check that gate checkpoint is v3.1+ and loaded (not random-init noop).~~ Fixed — Phase 1B/4.
5. No long-turn soak test **result** mirroring real chat (`CHAT_MAX_NEW_TOKENS=256`) — script exists (Phase 3E) but has not been run on GPU yet.
6. Microscope API and CLI diverge slightly in defaults (temperature, introspection) — not yet addressed (low priority, P2).

---

## Phase 1 — Runtime safety (P0)

**Track A: Collapse guard in `ChatEngine`**

- After `generate_reply`, run `detect_empathy_collapse(reply)` and `collapse_score(reply)` on the decoded new tokens (already isolated — no prompt leak).
- If collapse detected:
  - Retry once with hooks disabled (`hook_strength=0` or `_should_modulate()=False`).
  - If still collapsed, return a safe fallback string + flag `{"collapse_detected": true, "recovered": bool}`.
- Add `DEFAULT_REPETITION_PENALTY` confirmation in chat path (already set in `generate_reply` when > 1.0).

**Track B: Gate load verification**

- On `ChatEngine.__init__`, log and expose `gate_source`, `gate_version` from checkpoint metadata.
- Warn (stderr + session banner) if `gate_load.source != "trained"` or `gate_version` predates `v3.1_listener_ce_hardened`.
- Modal worker `setup()`: assert trained gate or fail fast with clear message.

**Tests:** `tests/test_chat_engine_collapse_guard.py` — mock generation returning glued morph text; assert retry/fallback.

---

## Phase 2 — Observability (P1)

**Track C: Per-turn session metrics**

- Extend `ChatSession.to_log_dict()` with per-turn:
  - `new_text`, `collapse_score`, `collapse_detected`
  - `affect_vector_norm`, `hook_strength`, `gate_output_norm` (‖gate(vec)‖)
  - `hooks_active` (bool)
- `/save` and Modal `save_session_artifact` write this schema (version field `chat_log_schema: 2`).

**Track D: CLI UX**

- Print collapse warning banner if guard fired (yellow `[collapse guard]` prefix).
- `/status` command: gate version, encoder/SNN sources, last turn metrics.
- Document commands in `chat.py` header and `data/README.md`.

**Tests:** round-trip `to_log_dict()` / load; Microscope API returns introspection fields.

---

## Phase 3 — Long-generation soak (P1)

**Track E: Chat soak benchmark**

- New `benchmark_phase_chat_soak.py` (Modal):
  - Drive `ChatEngine` (or worker) through 10-turn scripted distress arc (reuse scenario templates, not holdout IDs in training registry).
  - `max_new_tokens=CHAT_MAX_NEW_TOKENS` (256).
  - Assert no turn has `collapse_detected: true`.
  - Record mean empathy lexical score drift, affect norm trajectory.

**Track F: Regression in CI**

- Lightweight pytest (no GPU): replay saved golden `new_text` samples through collapse detector.
- Optional `@pytest.mark.slow` Modal job in `scripts/run_behavioral_verification.py --include-chat-soak`.

---

## Phase 4 — Modal worker production hygiene (P2)

**Track G: Worker lifecycle**

- `@modal.enter`: verify HF token, gate checkpoint mtime, log `GATE_VERSION`.
- Idle timeout / `@modal.concurrent` limits documented.
- `chat_turn` returns same schema as local `generate_reply` + collapse guard fields.

**Track H: Deploy path**

- Document one-liner: `py -3 chat.py --modal` (user-facing).
- Optional Gradio/Streamlit thin wrapper (`scripts/chat_ui.py`) — **defer until Phase 1–3 green**.

---

## Phase 5 — Microscope API alignment (P2)

- Route all generation through shared `ChatEngine.generate_reply(..., return_introspection=True)`.
- Expose collapse guard status in `/chat` JSON response.
- Add `/health` with gate version + checkpoint sources.
- Extend `tests/test_microscope_api.py` for collapse guard mock.

---

## Success criteria (exit checklist)

| Check | Target |
|-------|--------|
| 10-turn soak, 256 tokens/turn | 0 collapses |
| CLI `/save` log | Every turn has metrics; schema versioned |
| Untrained gate | Fail-fast or visible warning |
| pytest | All chat/microscope tests pass; soak benchmark in verification script |
| docs | `data/README.md` + this plan marked complete |

---

## Suggested implementation order

1. Phase 1A collapse guard + tests (1–2 days)
2. Phase 1B gate load verification (half day)
3. Phase 2 session metrics + `/status` (1 day)
4. Phase 3 soak benchmark + verification hook (1 day)
5. Phase 4–5 as needed for demo/deploy

---

## References

- Pipeline: `src/chat/engine.py`, `chat.py`, `run_chat.py`
- Collapse detection: `src/benchmark/gate_holdout.py`, `stats["new_text"]` in `src/llm/loader.py`
- Behavioral baseline: `docs/results.md` § Gate v3.1
- Benchmark parity: `benchmark_phase_chat_ab.py`
