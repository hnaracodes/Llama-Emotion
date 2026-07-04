"""Runtime path resolution tests."""

from pathlib import Path

from src.config import AFFECT_ENCODER_DIR, EMPATHETICDIALOGUES_DIR
from src.runtime_paths import (
    affect_encoder_dir,
    empatheticdialogues_dir,
    is_modal_runtime,
)


def test_local_paths_default(monkeypatch):
    monkeypatch.delenv("SAA_RUNTIME", raising=False)
    assert is_modal_runtime() is False
    assert empatheticdialogues_dir() == EMPATHETICDIALOGUES_DIR
    assert affect_encoder_dir() == AFFECT_ENCODER_DIR


def test_modal_paths_on_volume(monkeypatch):
    monkeypatch.setenv("SAA_RUNTIME", "modal")
    assert is_modal_runtime() is True
    assert empatheticdialogues_dir() == Path("/models/data/raw/empatheticdialogues")
    assert affect_encoder_dir() == Path("/models/affect")
