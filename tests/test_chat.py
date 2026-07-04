"""Tests for emotional CLI chat modules (no GPU)."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.chat.session import ChatMessage, ChatSession
from src.chat.signatures import compute_traits, ema_update
from src.chat.tone_markers import (
    detect_shift,
    dominant_tone as tone_from_traits,
    format_shift_banner,
    tone_prefix,
)
from src.chat.transcript import format_tribev2_transcript


def test_chat_session_append():
    s = ChatSession()
    s.append("user", "hello")
    assert len(s.messages) == 1
    assert s.messages[0].role == "user"


def test_tribev2_transcript_format():
    t0 = time.time()
    msgs = [
        ChatMessage("user", "hi", timestamp=t0),
        ChatMessage("assistant", "hello", timestamp=t0 + 45),
    ]
    text = format_tribev2_transcript(msgs, session_start=t0)
    assert "[00:00] user: hi" in text
    assert "[00:45] assistant: hello" in text


def test_compute_traits_and_shift():
    aff = np.random.randn(10, 32).astype(np.float32)
    spikes = (np.abs(np.diff(aff, axis=0, prepend=aff[:1])) > 0.1).astype(np.float32)
    traits = compute_traits(aff, spikes, {"mean_firing_rate": 0.2})
    assert "engagement" in traits
    assert 0.0 <= traits["engagement"] <= 1.0


def test_ema_update():
    a = np.zeros(32, dtype=np.float32)
    b = np.ones(32, dtype=np.float32)
    out = ema_update(a, b, alpha=0.5)
    assert np.allclose(out, 0.5)


def test_tone_shift_detection():
    before = {"engagement": 0.2, "arousal": 0.2, "tension": 0.2, "warmth": 0.2, "stability": 0.8, "shift": 0.0}
    after = {"engagement": 0.8, "arousal": 0.7, "tension": 0.5, "warmth": 0.7, "stability": 0.5, "shift": 0.3}
    shifted, mag, b, a = detect_shift(before, after, threshold=0.1)
    assert shifted
    assert mag > 0
    assert tone_prefix(a)


def test_shift_banner_contains_tones():
    banner = format_shift_banner("neutral", "warm", 0.21)
    assert "neutral" in banner
    assert "warm" in banner


def test_phase_chat_log_roundtrip(tmp_path):
    """Smoke: session log serializes for phase_chat.json."""
    s = ChatSession()
    s.append("user", "I failed my exam.")
    s.append("assistant", "That sounds hard.")
    s.traits = {"engagement": 0.7, "arousal": 0.4, "warmth": 0.6, "shift": 0.2, "tension": 0.3, "stability": 0.7}
    s.dominant_tone = tone_from_traits(s.traits)
    s.record_tone_event(
        event="refresh",
        before="neutral",
        after=s.dominant_tone,
        shift=0.21,
        traits=s.traits,
    )
    out = tmp_path / "phase_chat.json"
    out.write_text(json.dumps(s.to_log_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["messages"]) == 2
    assert len(data["tone_timeline"]) == 1


def test_to_log_dict_schema_v2_includes_turn_metrics(tmp_path):
    """Phase 2C (docs/chat_hardening_plan.md): schema-versioned per-turn
    collapse/affect diagnostics round-trip through to_log_dict()."""
    from src.config import CHAT_LOG_SCHEMA_VERSION

    s = ChatSession()
    s.append("user", "I lost my job today.")
    s.append("assistant", "That's a lot to carry.")
    s.turn_metrics.append(
        {
            "turn_index": 1,
            "new_text": "That's a lot to carry.",
            "collapse_detected": False,
            "collapse_score": 0.0,
            "recovered": False,
            "hooks_active": True,
            "hook_strength": 1.0,
            "affect_vector_norm": 0.42,
            "gate_output_norm": 0.05,
            "elapsed_sec": 1.23,
        }
    )
    out = tmp_path / "phase_chat.json"
    out.write_text(json.dumps(s.to_log_dict(), indent=2), encoding="utf-8")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chat_log_schema"] == CHAT_LOG_SCHEMA_VERSION == 2
    assert len(data["turn_metrics"]) == 1
    assert data["turn_metrics"][0]["collapse_detected"] is False
    assert data["turn_metrics"][0]["affect_vector_norm"] == 0.42


def test_run_tribev2_from_transcript_fallback():
    from src.affective.tribev2_client import run_tribev2_from_transcript

    msgs = [ChatMessage("user", "test message")]
    fmri, source = run_tribev2_from_transcript(msgs)
    assert fmri.ndim == 2
    assert "synthetic" in source or "tribev2" in source


def test_affect_dynamics_across_turns():
    from src.affective.dynamics import AffectDynamics
    from src.affective.emotion_lexicon import emotion_to_32d

    dyn = AffectDynamics(decay=0.8, gain=0.35)
    v1 = dyn.step(emotion_to_32d("anxious"))
    v2 = dyn.step(emotion_to_32d("joyful"))
    assert not np.allclose(v1, v2)
    assert len(dyn.trajectory) == 2


@patch("src.chat.engine.extract_signature_from_pipeline")
@patch("src.chat.engine.register_affective_hooks", return_value=[])
@patch("src.brain.checkpoints.load_gate")
@patch("src.brain.checkpoints.load_amygdala")
@patch("src.brain.checkpoints.load_encoder")
def test_multi_turn_affect_vectors_differ(mock_enc, mock_amy, mock_gate, _hooks, mock_sig):
    """Track A: consecutive refresh_affect calls evolve session affect (no GPU)."""
    import torch

    from src.brain.checkpoints import LoadResult
    from src.brain.lif_network import LIFAmygdala
    from src.affective.encoder import AffectEncoder
    from src.chat.engine import ChatEngine
    from src.config import AFFECT_DIM

    def _mock_model():
        model = MagicMock()
        model.config.hidden_size = 2048
        p = torch.nn.Parameter(torch.zeros(1))
        model.parameters = lambda: iter([p])
        return model

    def _mock_tokenizer():
        tok = MagicMock()
        tok.pad_token_id = 0
        return tok

    enc = AffectEncoder(backend="hash")
    amy = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    mock_enc.return_value = (enc, LoadResult(source="random_init"))
    mock_amy.return_value = (amy, LoadResult(source="random_init"))
    mock_gate.return_value = LoadResult(source="random_init")

    vec_a = np.random.randn(AFFECT_DIM).astype(np.float32)
    vec_b = np.random.randn(AFFECT_DIM).astype(np.float32)
    mock_sig.side_effect = [
        {"vector": vec_a, "traits": {"engagement": 0.5, "shift": 0.0}, "snn_mem_state": None},
        {"vector": vec_b, "traits": {"engagement": 0.6, "shift": 0.1}, "snn_mem_state": ("m1", "m2")},
    ]

    engine = ChatEngine(_mock_model(), _mock_tokenizer(), encoder_backend="hash")
    engine.session.append("user", "I failed my exam and feel devastated.")
    engine.refresh_affect(force=True)
    v1 = engine.session.affect_vector.copy()

    engine.session.append("user", "Actually I'm starting to feel hopeful.")
    engine.refresh_affect(force=True)
    v2 = engine.session.affect_vector.copy()

    assert v1.shape == (AFFECT_DIM,)
    assert v2.shape == (AFFECT_DIM,)
    assert not np.allclose(v1, v2)
    assert engine.session.affect_dynamics is not None
    assert len(engine.session.affect_dynamics.trajectory) == 2
    assert engine.session.snn_mem_state == ("m1", "m2")


def test_session_reset_affect_state_clears_membrane():
    s = ChatSession()
    s.snn_mem_state = ("a", "b")
    s.turn_index = 3
    s.affect_trajectory.append([0.0] * 32)
    s.reset_affect_state()
    assert s.snn_mem_state is None
    assert s.turn_index == 0
    assert s.affect_trajectory == []
