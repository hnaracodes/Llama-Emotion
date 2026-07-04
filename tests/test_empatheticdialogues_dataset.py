"""Tests for EmpatheticDialogues dataset + holdout registry."""

from pathlib import Path

import numpy as np
import pytest

from src.affective.dataset import (
    EmpatheticDialoguesDataset,
    HoldoutRegistry,
    ScenarioHoldoutDataset,
    collect_holdout_texts,
)
from src.affective.encoder import AffectEncoder
from src.affective.perception import estimate_user_affect
from src.config import PROJECT_ROOT

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "empatheticdialogues"


def test_fixture_train_loads():
    ds = EmpatheticDialoguesDataset("train", data_dir=FIXTURE_DIR, filter_holdouts=False)
    assert len(ds) >= 2
    sample = ds[0]
    assert sample.emotion
    assert len(sample.utterances) >= 1
    assert sample.target_32d.shape[0] == 32


def test_holdout_registry_catches_phase4_prompt():
    reg = HoldoutRegistry()
    texts = collect_holdout_texts()
    assert any("photosynthesis" in t for t in texts)


def test_train_filters_holdout_leak():
    ds = EmpatheticDialoguesDataset("train", data_dir=FIXTURE_DIR, filter_holdouts=True)
    reg = HoldoutRegistry()
    for s in ds:
        assert not reg.contains_holdout(s)


def test_scenario_holdout_dataset():
    ds = ScenarioHoldoutDataset()
    assert len(ds) >= 6  # 3 chat_ab + 5 phase4 + scenarios
    ids = {s.conv_id for s in ds}
    assert "scenario_conflict_escalation" in ids


def test_perception_empty_matches_pipeline():
    enc = AffectEncoder(backend="hash")
    vec, source = estimate_user_affect([], encoder=enc)
    assert source.startswith("encoder:")
    assert vec.shape == (32,)
    assert float(np.linalg.norm(vec)) > 0


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmpatheticDialoguesDataset("train", data_dir=tmp_path)
