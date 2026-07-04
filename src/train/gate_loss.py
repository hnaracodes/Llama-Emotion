"""Gate v3 losses: listener-reply sequence CE with hooks on/off."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import AFFECT_DIM, GATE_DISTRESS_MARGIN, GATE_NEUTRAL_CE_EPS
from src.llm.hooks import AffectiveGate, AffectiveState, register_affective_hooks


def listener_sequence_ce(
    model,
    tokenizer,
    context_messages: list[dict[str, str]],
    listener_reply: str,
    *,
    gate: AffectiveGate,
    affect_vector: torch.Tensor,
    hooks_on: bool,
    device: torch.device,
    max_target_tokens: int = 128,
) -> torch.Tensor:
    """Teacher-forced CE on human listener reply; hooks optional."""
    prompt_text = tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    target_ids = tokenizer(
        listener_reply,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if target_ids.shape[1] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if target_ids.shape[1] > max_target_tokens:
        target_ids = target_ids[:, :max_target_tokens]

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = torch.cat(
        [torch.full_like(prompt_ids, -100), target_ids],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)

    state = AffectiveState(AFFECT_DIM, device=str(device))
    state.set(affect_vector)
    handles: list = []
    if hooks_on:
        handles = register_affective_hooks(model, gate, state.get, strength=1.0)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    finally:
        for h in handles:
            h.remove()


def distress_margin_loss(
    ce_on: torch.Tensor,
    ce_off: torch.Tensor,
    margin: float = GATE_DISTRESS_MARGIN,
) -> torch.Tensor:
    """Hooks-on should lower CE vs hooks-off by at least margin."""
    return F.relu(margin - (ce_off.detach() - ce_on))


def neutral_noop_loss(
    ce_on: torch.Tensor,
    ce_off: torch.Tensor,
    eps: float = GATE_NEUTRAL_CE_EPS,
) -> torch.Tensor:
    """Neutral: hooks-on should not increase CE beyond hooks-off + eps.

    Note: this is deliberately the *only* term driving the neutral bucket.
    Unlike the distress bucket, neutral must not include a bare `ce_on`
    minimization term — that would reward the gate for generically helping
    predict the (neutral) listener reply regardless of the affect vector's
    content, which defeats the "hooks should be inert on neutral input"
    invariant this loss is supposed to encode.
    """
    return F.relu(ce_on - ce_off.detach() - eps)
