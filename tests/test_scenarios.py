"""Holdout scenario loading and registry tests."""

from src.affective.dataset import HoldoutRegistry, ScenarioHoldoutDataset, collect_holdout_texts
from src.benchmark.scenarios import load_holdout_scenarios


def test_holdout_scenario_count():
    scenarios = load_holdout_scenarios()
    assert len(scenarios) >= 25
    categories = {s.category for s in scenarios}
    assert "distress_recovery" in categories
    assert "conflict" in categories
    assert "factual_neutral" in categories
    assert "tone_shift" in categories


def test_tone_shift_scenarios_tagged():
    tone = load_holdout_scenarios(category="tone_shift")
    assert len(tone) >= 5
    assert all("sudden_tone_shift" in s.tags for s in tone)


def test_long_arcs_for_loop_benchmark():
    long_arcs = load_holdout_scenarios(min_turns=8)
    assert len(long_arcs) >= 4


def test_scenario_holdout_dataset_includes_files():
    ds = ScenarioHoldoutDataset()
    assert len(ds) >= 30  # 3 chat_ab + 5 phase4 + ~29 scenarios
    ids = {s.conv_id for s in ds}
    assert "scenario_tone_calm_to_panic" in ids
    assert "scenario_factual_week_planning" in ids


def test_holdout_texts_include_scenario_utterances():
    texts = collect_holdout_texts()
    assert any("firmware update" in t for t in texts)


def test_train_registry_blocks_scenario_leak():
    from pathlib import Path

    from src.affective.dataset import DialogueSample, EmpatheticDialoguesDataset
    from src.config import PROJECT_ROOT

    fixture = PROJECT_ROOT / "tests" / "fixtures" / "empatheticdialogues"
    reg = HoldoutRegistry()
    leak = DialogueSample(
        conv_id="leak",
        emotion="neutral",
        prompt="Is it going to rain this afternoon?",
        utterances=["Is it going to rain this afternoon?"],
        split="train",
        target_32d=None,
    )
    assert reg.contains_holdout(leak)
    ds = EmpatheticDialoguesDataset("train", data_dir=fixture, filter_holdouts=True)
    for s in ds:
        assert not reg.contains_holdout(s)
