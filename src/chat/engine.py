"""Chat inference engine: W4 Llama + encoder/SNN affect + neuromodulatory hooks."""

from __future__ import annotations

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
    CHAT_HOOK_STRENGTH,
    CHAT_KV_BITS,
    CHAT_MAX_HISTORY_TOKENS,
    CHAT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DELTA_THETA,
    GATE_NOOP_EPS,
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

    @torch.inference_mode()
    def generate_reply(
        self,
        user_text: str,
        *,
        max_new_tokens: int = CHAT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        return_introspection: bool = False,
    ) -> dict[str, Any]:
        self.session.append("user", user_text)
        messages = trim_messages_by_tokens(
            self.tokenizer,
            self.session.messages,
            CHAT_MAX_HISTORY_TOKENS,
        )
        self.session.messages = messages

        self.refresh_affect(force=True, messages=messages)

        prompt = build_llama_prompt(self.tokenizer, messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        handles: list = []
        introspection: dict[str, Any] | None = None
        if self._should_modulate():
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
        try:
            if return_introspection and self._should_modulate():
                from src.benchmark.affect_metrics import last_token_logits

                logits_on = last_token_logits(self.model, self.tokenizer, prompt)
            out = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if return_introspection and self._should_modulate():
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
        self.session.append("assistant", reply)

        result = {
            "reply": reply,
            "traits": dict(self.session.traits),
            "dominant_tone": self.session.dominant_tone,
            "elapsed_sec": elapsed,
            "tribe_source": self._last_affect_source,
            "affect_source": self._last_affect_source,
            "encoder_source": self._encoder_load.source,
            "gate_source": self._gate_load.source,
            "amygdala_source": self._amygdala_load.source,
        }
        if return_introspection:
            if introspection is None:
                vec = self.session.affect_vector
                introspection = {
                    "affect_vector_norm": float(np.linalg.norm(vec))
                    if vec is not None
                    else 0.0,
                    "hooks_on": False,
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
