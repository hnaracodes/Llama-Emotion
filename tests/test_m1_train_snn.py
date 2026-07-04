"""M1 — local SNN training smoke test."""

from pathlib import Path

from src.brain.checkpoints import load_amygdala
from src.config import SNN_CKPT_DIR


def test_train_snn_fixture_creates_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.SNN_CKPT_DIR", tmp_path)
    from train_snn import train_local

    result = train_local(fixture=True, max_samples=4, epochs=1)
    assert Path(result["checkpoint"]).is_file()
    assert result["tribev2_used"] is False
    model, load = load_amygdala(path=Path(result["checkpoint"]), input_dim=32)
    assert load.source == "trained"
