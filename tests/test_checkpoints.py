"""Checkpoint load/save tests."""

from pathlib import Path

import torch

from src.affective.encoder import AffectEncoder
from src.brain.checkpoints import assert_gate_noop, load_encoder, load_gate, save_gate
from src.config import AFFECT_DIM, MODEL_ID
from src.llm.hooks import AffectiveGate


def test_load_encoder_missing_returns_random():
    enc, res = load_encoder(path=Path("/nonexistent/encoder.pt"), backend="hash")
    assert res.source == "random_init"
    assert enc.encode_text("hi").shape == (AFFECT_DIM,)


def test_gate_noop_at_zero():
    gate = AffectiveGate(AFFECT_DIM, 128, mode="additive")
    assert_gate_noop(gate)


def test_save_load_gate_roundtrip(tmp_path):
    gate = AffectiveGate(AFFECT_DIM, 2048, mode="additive")
    path = save_gate(gate, tmp_path / "affect_gate.pt", model_id=MODEL_ID, hidden_size=2048)
    gate2 = AffectiveGate(AFFECT_DIM, 2048, mode="additive")
    res = load_gate(gate2, path=path, model_id=MODEL_ID, hidden_size=2048)
    assert res.source == "trained"


def test_load_amygdala_without_supervision_is_unverified(tmp_path):
    from src.brain.checkpoints import load_amygdala
    from src.brain.lif_network import LIFAmygdala

    model = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    path = tmp_path / "amygdala.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    loaded, res = load_amygdala(path=path, input_dim=AFFECT_DIM)
    assert res.source == "unverified_checkpoint"
    assert loaded is not None
