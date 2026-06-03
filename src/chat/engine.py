"""Chat inference engine: W4 Llama + hooks + affect refresh."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.chat.session import ChatSession
from src.chat.signatures import (
    apply_manual_affect_scale,
    ema_update,
    extract_signature_from_pipeline,
    trait_shift_magnitude,
)
from src.chat.transcript import build_llama_prompt, trim_messages_by_tokens
from src.chat.tone_markers import detect_shift, dominant_tone
from src.config import (
    AFFECT_DIM,
    AFFECT_REFRESH_SEC,
    CHAT_HOOK_STRENGTH,
    CHAT_KV_BITS,
    CHAT_MAX_HISTORY_TOKENS,
    CHAT_MAX_NEW_TOKENS,
    DELTA_THETA,
)
from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks


class ChatEngine:
    """Persistent chat with neuromodulatory hooks and periodic affect refresh."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        hook_strength: float = CHAT_HOOK_STRENGTH,
        kv_bits: int = CHAT_KV_BITS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_bits = kv_bits
        self.device = next(model.parameters()).device
        hidden_size = model.config.hidden_size

        self.gate = AffectiveGate(AFFECT_DIM, hidden_size, mode="additive").to(self.device)
        self.amygdala = None
        self.affect_state = AffectiveState(AFFECT_DIM, device=str(self.device))
        self.affect_state.zero()
        self.session = ChatSession(hook_strength=hook_strength)
        self._hook_handles: list = []
        self._last_tribe_source = ""
        self._ensure_hooks(hook_strength)

    def _ensure_hooks(self, strength: float) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = register_affective_hooks(
            self.model,
            self.gate,
            self.affect_state.get,
            strength=strength,
        )

    def set_hook_strength(self, strength: float) -> None:
        self.session.hook_strength = strength
        self._ensure_hooks(strength)

    def set_manual_affect(self, scale: float | None) -> None:
        """Override affect magnitude (/affect high=2.0, low=0.0, neutral=None)."""
        self.session.manual_affect_scale = scale
        self._sync_affect_state()

    def _sync_affect_state(self) -> None:
        vec = self.session.affect_vector
        if vec is None:
            self.affect_state.zero()
            return
        scaled = apply_manual_affect_scale(vec, self.session.manual_affect_scale)
        self.affect_state.set(
            torch.from_numpy(scaled).to(device=self.device, dtype=torch.float32)
        )

    @torch.inference_mode()
    def generate_reply(
        self,
        user_text: str,
        *,
        max_new_tokens: int = CHAT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        if self.session.needs_affect_refresh(AFFECT_REFRESH_SEC):
            self.refresh_affect(force=False)

        self.session.append("user", user_text)
        messages = trim_messages_by_tokens(
            self.tokenizer,
            self.session.messages,
            CHAT_MAX_HISTORY_TOKENS,
        )
        self.session.messages = messages

        prompt = build_llama_prompt(self.tokenizer, messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        elapsed = time.perf_counter() - t0

        new_ids = out[0, prompt_len:]
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        self.session.append("assistant", reply)

        return {
            "reply": reply,
            "traits": dict(self.session.traits),
            "dominant_tone": self.session.dominant_tone,
            "elapsed_sec": elapsed,
            "tribe_source": self._last_tribe_source,
        }

    def refresh_affect(self, *, force: bool = True) -> dict[str, Any]:
        """TRIBEv2/synthetic → pipeline → SNN → EMA update affect state."""
        from src.affective.tribev2_client import (
            pipeline_to_spikes,
            run_tribev2_from_transcript,
        )
        from src.brain.lif_network import LIFAmygdala

        msgs = self.session.transcript_messages()
        if not msgs:
            return {"ok": False, "reason": "empty transcript"}

        if not force and not self.session.needs_affect_refresh(AFFECT_REFRESH_SEC):
            return {"ok": False, "reason": "refresh not due"}

        prev_traits = dict(self.session.traits)
        prev_tone = self.session.dominant_tone
        prev_vector = self.session.affect_vector

        fmri, source = run_tribev2_from_transcript(msgs)
        self._last_tribe_source = source
        pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)

        if self.amygdala is None:
            self.amygdala = LIFAmygdala(input_dim=pipe["D"], output_dim=AFFECT_DIM).to(
                self.device
            )

        sig = extract_signature_from_pipeline(
            fmri,
            pipe,
            amygdala=self.amygdala,
            device=self.device,
            prev_vector=prev_vector,
        )
        new_vec = ema_update(prev_vector, sig["vector"])
        self.session.affect_vector = new_vec
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
            "source": source,
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
