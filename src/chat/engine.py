"""Chat inference engine: W4 Llama + encoder/SNN affect + neuromodulatory hooks."""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.chat.session import ChatSession
from src.chat.signatures import (
    apply_manual_affect_scale,
    extract_signature_from_pipeline,
    trait_shift_magnitude,
)
from src.chat.transcript import build_llama_prompt, trim_messages_by_tokens
from src.chat.tone_markers import detect_shift, dominant_tone
from src.config import (
    AFFECT_DIM,
    AFFECT_ENCODER_BACKEND,
    CHAT_COLLAPSE_FALLBACK_REPLY,
    CHAT_HOOK_STRENGTH,
    CHAT_KV_BITS,
    CHAT_MAX_HISTORY_TOKENS,
    CHAT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DELTA_THETA,
    GATE_NOOP_EPS,
    GATE_VERSION,
    MODEL_ID,
)
from src.affective.affect_norm import clip_affect_norm
from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks


class ChatEngine:
    """Persistent chat with neuromodulatory hooks and per-turn affect refresh (M1)."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        hook_strength: float = CHAT_HOOK_STRENGTH,
        kv_bits: int = CHAT_KV_BITS,
        use_tribev2: bool = False,
        encoder_backend: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_bits = kv_bits
        self.use_tribev2 = use_tribev2
        self.device = next(model.parameters()).device
        hidden_size = model.config.hidden_size

        from src.brain.checkpoints import load_amygdala, load_encoder, load_gate

        self.encoder, self._encoder_load = load_encoder(
            backend=encoder_backend or AFFECT_ENCODER_BACKEND,
            device="cpu",
        )
        self.amygdala, self._amygdala_load = load_amygdala(
            input_dim=AFFECT_DIM,
            device=str(self.device),
        )
        self.gate = AffectiveGate(AFFECT_DIM, hidden_size, mode="additive").to(self.device)
        self._gate_load = load_gate(
            self.gate,
            model_id=MODEL_ID,
            hidden_size=hidden_size,
            device=str(self.device),
        )
        self.affect_state = AffectiveState(AFFECT_DIM, device=str(self.device))
        self.affect_state.zero()
        self.session = ChatSession(hook_strength=hook_strength)
        self._hook_handles: list = []
        self._last_affect_source = ""

        # Phase 1B (docs/chat_hardening_plan.md): surface gate provenance so
        # a random-init or stale-version gate can't silently masquerade as a
        # working affective adapter in an interactive session.
        self.gate_version: str | None = (
            (self._gate_load.meta or {}).get("gate_version")
            if self._gate_load.meta
            else None
        )
        self.gate_healthy = (
            self._gate_load.source == "trained" and self.gate_version == GATE_VERSION
        )
        warning = self.gate_health().get("warning")
        if warning:
            print(f"[chat] WARNING: {warning}", file=sys.stderr)

    def gate_health(self) -> dict[str, Any]:
        """Report gate checkpoint provenance for CLI /status and health checks."""
        source = self._gate_load.source
        warning: str | None = None
        if source != "trained":
            warning = (
                f"gate checkpoint source={source!r} — no trained gate found; "
                "hooks will inject an untrained (near-random) bias. Train "
                "with `py -3 -m modal run train_gate.py` first."
            )
        elif self.gate_version != GATE_VERSION:
            warning = (
                f"gate checkpoint version={self.gate_version!r} does not match "
                f"current config.GATE_VERSION={GATE_VERSION!r} — this checkpoint "
                "predates or postdates the hardened v3.1 training/eval fixes; "
                "behavior may not match docs/results.md."
            )
        return {
            "source": source,
            "version": self.gate_version,
            "expected_version": GATE_VERSION,
            "healthy": self.gate_healthy,
            "warning": warning,
        }

    def _should_modulate(self) -> bool:
        """AF-4: hooks only when strength > 0 and affect vector is non-neutral."""
        if self.session.hook_strength <= 0:
            return False
        vec = self.session.affect_vector
        if vec is None:
            return False
        return float(np.linalg.norm(vec)) > GATE_NOOP_EPS

    def _ensure_hooks(self, strength: float) -> None:
        """Legacy helper — prefer per-generation registration in generate_reply."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        if strength > 0:
            self._hook_handles = register_affective_hooks(
                self.model,
                self.gate,
                self.affect_state.get,
                strength=strength,
            )

    def set_hook_strength(self, strength: float) -> None:
        self.session.hook_strength = strength
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def set_manual_affect(self, scale: float | None) -> None:
        self.session.manual_affect_scale = scale
        self._sync_affect_state()

    def _sync_affect_state(self) -> None:
        vec = self.session.affect_vector
        if vec is None:
            self.affect_state.zero()
            return
        scaled = apply_manual_affect_scale(vec, self.session.manual_affect_scale)
        clipped = clip_affect_norm(np.asarray(scaled, dtype=np.float32))
        self.affect_state.set(
            torch.from_numpy(np.asarray(clipped)).to(device=self.device, dtype=torch.float32)
        )

    def _generate_once(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        hooks_enabled: bool,
        want_introspection: bool,
    ) -> dict[str, Any]:
        """Single generation pass; decodes new tokens only (no prompt leak)."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        handles: list = []
        if hooks_enabled:
            handles = register_affective_hooks(
                self.model,
                self.gate,
                self.affect_state.get,
                strength=self.session.hook_strength,
            )

        t0 = time.perf_counter()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if DEFAULT_REPETITION_PENALTY > 1.0:
            gen_kwargs["repetition_penalty"] = DEFAULT_REPETITION_PENALTY

        introspection: dict[str, Any] | None = None
        try:
            if want_introspection and hooks_enabled:
                from src.benchmark.affect_metrics import last_token_logits

                logits_on = last_token_logits(self.model, self.tokenizer, prompt)
            out = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if want_introspection and hooks_enabled:
                from src.benchmark.affect_metrics import (
                    kl_divergence_from_logits,
                    last_token_logits,
                )

                for h in handles:
                    h.remove()
                handles = []
                logits_off = last_token_logits(self.model, self.tokenizer, prompt)
                rolling_kl = kl_divergence_from_logits(logits_off, logits_on)
                vec = self.session.affect_vector
                introspection = {
                    "affect_vector_norm": float(np.linalg.norm(vec))
                    if vec is not None
                    else 0.0,
                    "hooks_on": True,
                    "rolling_kl_vs_hooks_off": rolling_kl,
                    "turn_index": self.session.turn_index,
                    "effective_affect_norm": float(np.linalg.norm(self.affect_state.get().cpu())),
                }
            for h in handles:
                h.remove()
        elapsed = time.perf_counter() - t0

        new_ids = out[0, prompt_len:]
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return {
            "reply": reply,
            "elapsed_sec": elapsed,
            "hooks_active": hooks_enabled,
            "introspection": introspection,
        }

    def _gate_output_norm(self) -> float:
        vec = self.affect_state.get()
        if vec is None:
            return 0.0
        with torch.no_grad():
            return float(self.gate(vec).norm().item())

    @torch.inference_mode()
    def generate_reply(
        self,
        user_text: str,
        *,
        max_new_tokens: int = CHAT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        return_introspection: bool = False,
    ) -> dict[str, Any]:
        from src.benchmark.gate_holdout import collapse_score, detect_empathy_collapse

        self.session.append("user", user_text)
        messages = trim_messages_by_tokens(
            self.tokenizer,
            self.session.messages,
            CHAT_MAX_HISTORY_TOKENS,
        )
        self.session.messages = messages

        self.refresh_affect(force=True, messages=messages)

        prompt = build_llama_prompt(self.tokenizer, messages, add_generation_prompt=True)
        hooks_enabled = self._should_modulate()

        # Phase 1A collapse guard (docs/chat_hardening_plan.md): chat allows
        # up to CHAT_MAX_NEW_TOKENS=256, well past the 64-96 tokens exercised
        # in training holdout, so a runtime check — not just the offline
        # benchmark suite — is required before trusting live output.
        attempt = self._generate_once(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            hooks_enabled=hooks_enabled,
            want_introspection=return_introspection,
        )
        reply = attempt["reply"]
        score = collapse_score(reply)
        collapsed = detect_empathy_collapse(reply)
        recovered = False

        if collapsed and attempt["hooks_active"]:
            # Retry once with hooks disabled — if the collapse only happens
            # with affective modulation on, an inert reply is far better
            # than a repetition loop.
            retry = self._generate_once(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                hooks_enabled=False,
                want_introspection=return_introspection,
            )
            if not detect_empathy_collapse(retry["reply"]):
                attempt = retry
                reply = retry["reply"]
                score = collapse_score(reply)
                collapsed = False
                recovered = True

        if collapsed:
            # Either hooks were already off (collapse is not gate-related) or
            # the hooks-off retry also collapsed — never surface raw looping
            # text to the user.
            reply = CHAT_COLLAPSE_FALLBACK_REPLY
            score = 0.0

        self.session.append("assistant", reply)

        vec = self.session.affect_vector
        affect_norm = float(np.linalg.norm(vec)) if vec is not None else 0.0
        turn_metrics = {
            "turn_index": self.session.turn_index,
            "new_text": reply,
            "collapse_detected": collapsed,
            "collapse_score": round(score, 4),
            "recovered": recovered,
            "hooks_active": attempt["hooks_active"],
            "hook_strength": self.session.hook_strength,
            "affect_vector_norm": round(affect_norm, 4),
            "gate_output_norm": round(self._gate_output_norm(), 4),
            "elapsed_sec": attempt["elapsed_sec"],
        }
        self.session.turn_metrics.append(turn_metrics)

        result = {
            "reply": reply,
            "traits": dict(self.session.traits),
            "dominant_tone": self.session.dominant_tone,
            "elapsed_sec": attempt["elapsed_sec"],
            "tribe_source": self._last_affect_source,
            "affect_source": self._last_affect_source,
            "encoder_source": self._encoder_load.source,
            "gate_source": self._gate_load.source,
            "gate_version": self.gate_version,
            "gate_healthy": self.gate_healthy,
            "amygdala_source": self._amygdala_load.source,
            "collapse_detected": collapsed,
            "collapse_score": round(score, 4),
            "recovered": recovered,
        }
        if return_introspection:
            introspection = attempt["introspection"]
            if introspection is None:
                introspection = {
                    "affect_vector_norm": affect_norm,
                    "hooks_on": attempt["hooks_active"],
                    "rolling_kl_vs_hooks_off": 0.0,
                    "turn_index": self.session.turn_index,
                }
            result["introspection"] = introspection
        return result

    def refresh_affect(
        self,
        *,
        force: bool = True,
        messages=None,
    ) -> dict[str, Any]:
        msgs = messages if messages is not None else self.session.transcript_messages()
        if not msgs:
            return {"ok": False, "reason": "empty transcript"}

        prev_traits = dict(self.session.traits)
        prev_tone = self.session.dominant_tone
        prev_vector = self.session.affect_vector

        if self.use_tribev2:
            from src.affective.tribev2_client import (
                pipeline_to_spikes,
                run_tribev2_from_transcript,
            )

            fmri, source = run_tribev2_from_transcript(msgs)
            self._last_affect_source = source
            pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
            fmri_ts = fmri
        else:
            from src.affective.pipeline import run_encoder_pipeline

            pipe, source = run_encoder_pipeline(
                msgs,
                encoder=self.encoder,
                source_prefix="encoder:empatheticdialogues_v1",
            )
            self._last_affect_source = source
            fmri_ts = np.zeros((pipe["T"], 1), dtype=np.float32)

        sig = extract_signature_from_pipeline(
            fmri_ts,
            pipe,
            amygdala=self.amygdala,
            device=self.device,
            prev_vector=prev_vector,
            snn_mem_state=self.session.snn_mem_state,
        )
        new_vec = sig["vector"]
        self.session.snn_mem_state = sig.get("snn_mem_state")

        from src.config import AFFECT_MEMBRANE_RESET_TURNS

        if (
            self.session.turn_index > 0
            and self.session.turn_index % AFFECT_MEMBRANE_RESET_TURNS == 0
        ):
            self.amygdala.reset_state()
            self.session.snn_mem_state = None
            if self.session.affect_dynamics is not None:
                self.session.affect_dynamics.reset()

        if prev_vector is not None:
            from src.affective.coupling import couple
            from src.affective.dynamics import AffectDynamics

            if self.session.affect_dynamics is None:
                self.session.affect_dynamics = AffectDynamics()
            coupled = couple(new_vec, prev_vector, coupling=0.35)
            new_vec = self.session.affect_dynamics.step(coupled)
        elif self.session.affect_dynamics is None:
            from src.affective.dynamics import AffectDynamics

            self.session.affect_dynamics = AffectDynamics()
            new_vec = self.session.affect_dynamics.step(new_vec)

        new_vec = clip_affect_norm(np.asarray(new_vec, dtype=np.float32))
        self.session.affect_vector = new_vec
        if self.session.affect_vector is not None:
            self.session.affect_trajectory.append(
                self.session.affect_vector.tolist()
            )
        self.session.traits = sig["traits"]
        self.session.traits["shift"] = round(
            float(np.linalg.norm(new_vec - prev_vector))
            if prev_vector is not None
            else sig["traits"].get("shift", 0.0),
            4,
        )
        self.session.dominant_tone = dominant_tone(self.session.traits)
        self.session.last_refresh_ts = time.time()
        self._sync_affect_state()

        magnitude = trait_shift_magnitude(prev_traits, self.session.traits)
        shifted, _, before_tone, after_tone = detect_shift(prev_traits, self.session.traits)

        if shifted:
            self.session.record_tone_event(
                event="refresh",
                before=before_tone if prev_traits else "neutral",
                after=after_tone,
                shift=magnitude,
                traits=self.session.traits,
            )

        return {
            "ok": True,
            "source": self._last_affect_source,
            "traits": dict(self.session.traits),
            "dominant_tone": self.session.dominant_tone,
            "shifted": shifted,
            "shift_magnitude": magnitude,
            "before_tone": before_tone if prev_traits else "neutral",
            "after_tone": after_tone,
        }

    def cleanup(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
