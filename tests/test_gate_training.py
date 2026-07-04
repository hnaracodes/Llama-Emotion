"""M1 — gate checkpoint noop and optional fixture training."""

from pathlib import Path

import pytest
import torch

from src.brain.checkpoints import assert_gate_noop, load_gate, save_gate
from src.config import AFFECT_DIM, GATE_VERSION, MODEL_ID
from src.llm.hooks import AffectiveGate
from src.train.gate import _gate_step_loss, _is_new_best_checkpoint, _load_frozen_llama


def test_random_gate_is_noop_at_zero():
    gate = AffectiveGate(AFFECT_DIM, 2048, mode="additive")
    assert_gate_noop(gate)


# --- v3.1 hardening: best-checkpoint tie-break -----------------------------


def test_best_checkpoint_ties_prefer_later_step():
    """A later eval tying the current best score must still win.

    Regression for a bug where strict '<' froze checkpoint selection on the
    *first* step to reach the eventual floor score, silently discarding all
    further training even though it kept running cleanly.
    """
    assert _is_new_best_checkpoint(0.0, float("inf")) is True
    assert _is_new_best_checkpoint(0.0, 0.0) is True  # tie -> later step wins
    assert _is_new_best_checkpoint(0.1, 0.0) is False  # strictly worse -> keep old best
    assert _is_new_best_checkpoint(0.0, 0.2) is True  # strictly better -> update


def test_best_checkpoint_selection_over_a_training_run():
    """Simulate a holdout history and confirm the *last* zero-score step wins."""
    scores = [0.8, 0.4, 0.0, 0.0, 0.0, 0.0]  # ties from step index 2 onward
    best_score = float("inf")
    best_step = -1
    for step, score in enumerate(scores):
        if _is_new_best_checkpoint(score, best_score):
            best_score = score
            best_step = step
    assert best_score == 0.0
    assert best_step == 5  # last tie, not the first (index 2)


# --- v3.1 hardening: neutral bucket must not reward bare ce_on ------------


def test_neutral_loss_does_not_reward_lower_ce_on():
    """Neutral bucket must not directly minimize ce_on (that would reward
    hooks-on for generically helping prediction regardless of affect
    content, defeating the 'hooks are inert on neutral input' invariant).
    """
    ce_on = torch.tensor(0.1)
    ce_off = torch.tensor(2.0)
    loss = _gate_step_loss("neutral", ce_on, ce_off)
    assert float(loss.item()) == 0.0


def test_distress_loss_rewards_lower_ce_on():
    """Distress bucket *should* directly reward hooks-on for lowering ce_on
    (behavior-cloning the listener reply) — this is the intended asymmetry
    with the neutral bucket above.
    """
    ce_on = torch.tensor(0.1)
    ce_off = torch.tensor(2.0)
    loss = _gate_step_loss("distress", ce_on, ce_off)
    assert float(loss.item()) == pytest.approx(0.1, abs=1e-4)


def test_neutral_loss_penalizes_hooks_hurting_prediction():
    ce_on = torch.tensor(1.5)
    ce_off = torch.tensor(1.0)
    loss = _gate_step_loss("neutral", ce_on, ce_off)
    assert float(loss.item()) > 0.0


def test_save_gate_records_v3_metadata(tmp_path):
    gate = AffectiveGate(AFFECT_DIM, 128, mode="additive")
    path = save_gate(
        gate,
        tmp_path / "affect_gate.pt",
        model_id=MODEL_ID,
        hidden_size=128,
        extra={"gate_version": GATE_VERSION},
    )
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["gate_version"] == GATE_VERSION


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Gate training requires CUDA")
def test_train_gate_fixture_creates_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.GATE_CKPT_DIR", tmp_path)
    from train_gate import train_local

    result = train_local(fixture=True, max_samples=2, epochs=1)
    path = Path(result["checkpoint"])
    assert path.is_file()
    assert result["gate_version"] == GATE_VERSION
    assert "best_step" in result
    assert "total_steps" in result
    assert result["best_step"] <= result["total_steps"]
    gate = AffectiveGate(AFFECT_DIM, 2048, mode="additive")
    load_gate(gate, path=path, model_id=MODEL_ID, hidden_size=2048)
    assert_gate_noop(gate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Gate training requires CUDA")
def test_load_frozen_llama_has_no_trainable_params():
    """Backward through the frozen Llama should never leave stray grads on
    its parameters; every Llama param must have requires_grad=False before
    the training loop starts (bug found by business-logic-auditor review —
    non-quantized params like embeddings/layernorms/lm_head default to
    requires_grad=True and were never explicitly frozen).
    """
    model, _ = _load_frozen_llama()
    assert not any(p.requires_grad for p in model.parameters())
