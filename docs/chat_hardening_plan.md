# Interactive chat hardening plan

Gate v3.1 is **collapse-free in benchmarks**, but the live chat path (`chat.py`, `run_chat.py`, `ChatEngine`, Microscope API) has not yet been held to the same standard. This plan closes that gap.

**Goal:** A user can run multi-turn emotional chat (local CUDA or Modal worker) with confidence that (1) generation will not collapse into empathy-token loops, (2) affect modulation is observable and controllable, and (3) failures are detected and surfaced instead of silently degrading.

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

**Gaps:**

1. No runtime `detect_empathy_collapse` on chat replies (256-token generations).
2. No automatic backoff (lower strength / disable hooks) when collapse is detected mid-session.
3. No structured session log with per-turn `new_text`, collapse score, hook strength, affect norm.
4. Modal worker has no health check that gate checkpoint is v3.1+ and loaded (not random-init noop).
5. No long-turn soak test mirroring real chat (`CHAT_MAX_NEW_TOKENS=256`).
6. Microscope API and CLI diverge slightly in defaults (temperature, introspection).

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
