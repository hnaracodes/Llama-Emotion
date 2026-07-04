"""Affective gate training loop (v3: listener CE + SNN-aligned vectors)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.config import (
    AFFECT_DIM,
    AFFECT_ENCODER_BACKEND,
    DEFAULT_REPETITION_PENALTY,
    GATE_DISTRESS_EMOTIONS,
    GATE_DISTRESS_MARGIN,
    GATE_EMPATHY_ID_WEIGHT,
    GATE_HOLDOUT_EVERY,
    GATE_HOLDOUT_MAX_NEW_TOKENS,
    GATE_NEUTRAL_BATCH_RATIO,
    GATE_NEUTRAL_CE_EPS,
    GATE_NEUTRAL_EMOTIONS,
    GATE_TRAIN_EPOCHS,
    GATE_TRAIN_MAX_SAMPLES,
    GATE_V3_LISTENER_MAX_TOKENS,
    GATE_VERSION,
    MODEL_ID,
    SUPERVISION_VERSION,
)
from src.train.gate_loss import (
    distress_margin_loss,
    listener_sequence_ce,
    neutral_noop_loss,
)
from src.train.gate_vector import build_gate_affect_vector


def _gate_step_loss(
    bucket: str,
    ce_on: torch.Tensor,
    ce_off: torch.Tensor,
) -> torch.Tensor:
    """Compose the per-step loss for a distress or neutral training sample.

    Distress: directly reward hooks-on for behavior-cloning the human
    listener reply (bare `ce_on`), plus a margin ensuring hooks help vs.
    hooks-off. Neutral: reward *only* via `neutral_noop_loss` — no bare
    `ce_on` term — so the gate isn't rewarded for generically improving
    predictions on neutral text regardless of the affect vector's content.
    """
    if bucket == "distress":
        loss = ce_on + distress_margin_loss(ce_on, ce_off, GATE_DISTRESS_MARGIN)
    else:
        loss = neutral_noop_loss(ce_on, ce_off, GATE_NEUTRAL_CE_EPS)
    if GATE_EMPATHY_ID_WEIGHT > 0:
        loss = loss + GATE_EMPATHY_ID_WEIGHT * ce_on
    return loss


def _load_frozen_llama():
    """Load Llama and explicitly freeze every parameter.

    bitsandbytes 4-bit weights aren't trainable by construction, but
    non-quantized params (embeddings, layernorms, lm_head) default to
    `requires_grad=True` and would otherwise accumulate unused gradients on
    every `loss.backward()` call through the hooked forward pass, wasting
    VRAM for the whole run since only `gate.parameters()` are ever optimized.
    """
    from src.llm.loader import load_quantized_llama

    model, tokenizer = load_quantized_llama()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def _is_new_best_checkpoint(score: float, best_score: float) -> bool:
    """True if `score` should replace `best_score` as the saved checkpoint.

    Uses `<=` (not `<`) so that among evals tied at the same collapse score,
    the *later* (more-trained) checkpoint wins instead of freezing on the
    first tie — a strict `<` would silently pin the saved gate to whichever
    step first reached the eventual floor score, discarding all further
    training even when it kept running cleanly.
    """
    return score <= best_score


def _build_training_schedule(
    distress_indices: list[int],
    neutral_indices: list[int],
    max_samples: int,
    neutral_ratio: float,
) -> list[tuple[int, str]]:
    n_neutral = int(max_samples * neutral_ratio)
    n_distress = max(1, max_samples - n_neutral)
    d = list(distress_indices)
    n = list(neutral_indices)
    np.random.shuffle(d)
    np.random.shuffle(n)
    d = d[: min(len(d), n_distress)]
    n = n[: min(len(n), n_neutral)]
    schedule: list[tuple[int, str]] = [(i, "distress") for i in d] + [
        (i, "neutral") for i in n
    ]
    np.random.shuffle(schedule)
    return schedule


def _eval_holdout_gate(
    model,
    tokenizer,
    gate,
    amygdala,
    encoder,
    *,
    device: torch.device,
) -> dict:
    from src.benchmark.gate_holdout import (
        collapse_score,
        detect_empathy_collapse,
        holdout_prompts,
        summarize_holdout_eval,
    )
    from src.benchmark.hybrid_runner import generate_with_affect

    ref_msgs = [{"role": "user", "content": "I feel anxious and need support."}]
    aff_high, _, _ = build_gate_affect_vector(
        ref_msgs, encoder=encoder, amygdala=amygdala, device=device
    )
    with torch.no_grad():
        gate_output_norm = float(gate(aff_high).norm().item())

    rows = []
    for entry in holdout_prompts():
        prompt = entry["prompt"]
        hooks_on_text, hooks_on_stats = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=aff_high,
            gate=gate,
            strength=1.0,
            hooks_enabled=True,
            max_new_tokens=GATE_HOLDOUT_MAX_NEW_TOKENS,
            temperature=0.0,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
        )
        hooks_off_text, _ = generate_with_affect(
            model,
            tokenizer,
            prompt,
            affect_vector=aff_high,
            gate=gate,
            strength=1.0,
            hooks_enabled=False,
            max_new_tokens=GATE_HOLDOUT_MAX_NEW_TOKENS,
            temperature=0.0,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
        )
        # Collapse detection must run on the newly generated continuation
        # only, not the full prompt+generation text — the prompt itself can
        # legitimately contain empathy-related words that would otherwise
        # inflate the detector's word-frequency heuristics.
        generated_only = hooks_on_stats.get("new_text", hooks_on_text)
        score = collapse_score(generated_only)
        rows.append(
            {
                "prompt_id": entry["id"],
                "collapse_detected": detect_empathy_collapse(generated_only),
                "collapse_score": score,
                "hooks_on_preview": hooks_on_text[-200:],
                "hooks_off_preview": hooks_off_text[-200:],
                # If hooks never change output text across every holdout eval,
                # the gate has likely collapsed to a trivial no-op — "no
                # collapse" alone can't distinguish that from real, working
                # affect modulation.
                "text_changed": hooks_on_text != hooks_off_text,
            }
        )
    summary = summarize_holdout_eval(rows)
    summary["gate_output_norm"] = gate_output_norm
    return summary


def train_gate_loop(
    *,
    data_dir: Path,
    out_dir: Path,
    max_samples: int = GATE_TRAIN_MAX_SAMPLES,
    epochs: int = GATE_TRAIN_EPOCHS,
    lr: float = 1e-4,
    backend: str | None = None,
) -> dict:
    from src.affective.dataset import EmpatheticDialoguesDataset
    from src.brain.checkpoints import load_amygdala, load_encoder, save_gate
    from src.llm.hooks import AffectiveGate

    if not torch.cuda.is_available():
        raise RuntimeError("Gate training requires CUDA for W4 Llama.")

    encoder, enc_meta = load_encoder(backend=backend or AFFECT_ENCODER_BACKEND)
    model, tokenizer = _load_frozen_llama()
    device = next(model.parameters()).device
    hidden = model.config.hidden_size

    amygdala, amy_meta = load_amygdala(device=str(device))
    amygdala.eval()
    for p in amygdala.parameters():
        p.requires_grad = False

    gate = AffectiveGate(AFFECT_DIM, hidden, mode="additive").to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)

    train_ds = EmpatheticDialoguesDataset("train", data_dir=data_dir)
    distress_indices = [
        i
        for i, s in enumerate(train_ds.samples)
        if s.emotion in GATE_DISTRESS_EMOTIONS and s.gate_context_and_listener()
    ]
    neutral_indices = [
        i
        for i, s in enumerate(train_ds.samples)
        if s.emotion in GATE_NEUTRAL_EMOTIONS and s.gate_context_and_listener()
    ]
    schedule = _build_training_schedule(
        distress_indices,
        neutral_indices,
        max_samples,
        GATE_NEUTRAL_BATCH_RATIO,
    )
    if not schedule:
        raise RuntimeError("No gate training samples with listener replies.")

    history = []
    holdout_history = []
    consecutive_collapse = 0
    best_ckpt_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
    best_holdout_score = float("inf")
    best_step = 0
    stopped_early = False

    global_step = 0
    for epoch in range(epochs):
        np.random.shuffle(schedule)
        total = 0.0
        n = 0
        gate.train()
        model.eval()

        for step, (idx, bucket) in enumerate(schedule):
            if step > 0 and step % 50 == 0:
                print(
                    f"gate epoch {epoch + 1}/{epochs} sample {step}/{len(schedule)}",
                    flush=True,
                )

            sample = train_ds[idx]
            pair = sample.gate_context_and_listener()
            if pair is None:
                continue
            context_msgs, listener_reply = pair

            affect_vec, _, _ = build_gate_affect_vector(
                context_msgs,
                encoder=encoder,
                amygdala=amygdala,
                device=device,
            )

            ce_on = listener_sequence_ce(
                model,
                tokenizer,
                context_msgs,
                listener_reply,
                gate=gate,
                affect_vector=affect_vec,
                hooks_on=True,
                device=device,
                max_target_tokens=GATE_V3_LISTENER_MAX_TOKENS,
            )
            with torch.no_grad():
                ce_off = listener_sequence_ce(
                    model,
                    tokenizer,
                    context_msgs,
                    listener_reply,
                    gate=gate,
                    affect_vector=affect_vec,
                    hooks_on=False,
                    device=device,
                    max_target_tokens=GATE_V3_LISTENER_MAX_TOKENS,
                )

            loss = _gate_step_loss(bucket, ce_on, ce_off)

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                gate.proj.bias.zero_()

            total += float(loss.item())
            n += 1
            global_step += 1

            if GATE_HOLDOUT_EVERY > 0 and global_step % GATE_HOLDOUT_EVERY == 0:
                gate.eval()
                holdout = _eval_holdout_gate(
                    model,
                    tokenizer,
                    gate,
                    amygdala,
                    encoder,
                    device=device,
                )
                holdout_history.append(
                    {"epoch": epoch, "step": global_step, **holdout}
                )
                holdout_score = float(holdout["max_collapse_score"])
                if _is_new_best_checkpoint(holdout_score, best_holdout_score):
                    best_holdout_score = holdout_score
                    best_step = global_step
                    best_ckpt_state = {
                        k: v.cpu().clone() for k, v in gate.state_dict().items()
                    }
                    consecutive_collapse = 0
                elif holdout.get("any_collapse"):
                    consecutive_collapse += 1
                else:
                    consecutive_collapse = 0

                if consecutive_collapse >= 2:
                    print("gate early-stop: holdout collapse detected", flush=True)
                    stopped_early = True
                    break
                gate.train()

        history.append({"epoch": epoch, "loss": total / max(n, 1)})
        if stopped_early:
            break

    if best_ckpt_state is not None:
        gate.load_state_dict({k: v.to(device) for k, v in best_ckpt_state.items()})

    ckpt = save_gate(
        gate,
        out_dir / "affect_gate.pt",
        model_id=MODEL_ID,
        hidden_size=hidden,
        extra={
            "supervision": SUPERVISION_VERSION,
            "encoder_source": enc_meta.source,
            "amygdala_source": amy_meta.source,
            "gate_version": GATE_VERSION,
        },
    )
    result = {
        "supervision": SUPERVISION_VERSION,
        "gate_version": GATE_VERSION,
        "tribev2_used": False,
        "encoder_source": enc_meta.source,
        "amygdala_source": amy_meta.source,
        "checkpoint": str(ckpt),
        "distress_samples": len(distress_indices),
        "neutral_samples": len(neutral_indices),
        "schedule_len": len(schedule),
        "stopped_early": stopped_early,
        "best_holdout_score": best_holdout_score,
        "best_step": best_step,
        "total_steps": global_step,
        "history": history,
        "holdout_history": holdout_history,
    }
    (out_dir / "train_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if holdout_history:
        (out_dir / "holdout_eval.json").write_text(
            json.dumps(holdout_history[-1], indent=2), encoding="utf-8"
        )
    return result
