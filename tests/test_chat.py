"""Tests for emotional CLI chat modules (no GPU)."""

import json
import time
from pathlib import Path

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


def test_run_tribev2_from_transcript_fallback():
    from src.affective.tribev2_client import run_tribev2_from_transcript

    msgs = [ChatMessage("user", "test message")]
    fmri, source = run_tribev2_from_transcript(msgs)
    assert fmri.ndim == 2
    assert "synthetic" in source or "tribev2" in source
