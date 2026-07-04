"""Track A — affect dynamics tests."""

import numpy as np

from src.affective.dynamics import AffectDynamics
from src.affective.emotion_lexicon import emotion_to_32d
from src.config import AFFECT_DIM


def test_dynamics_decay_toward_recent():
    dyn = AffectDynamics(decay=0.5, gain=1.0)
    v1 = emotion_to_32d("anxious")
    v2 = emotion_to_32d("neutral")
    s1 = dyn.step(v1)
    s2 = dyn.step(v2)
    assert np.linalg.norm(s2) < np.linalg.norm(s1) + 0.5


def test_trajectory_length():
    dyn = AffectDynamics()
    for _ in range(3):
        dyn.step(emotion_to_32d("grateful"))
    assert len(dyn.trajectory) == 3


def test_repeated_zero_input_decays_toward_zero():
    dyn = AffectDynamics(decay=0.85, gain=0.35)
    dyn.step(emotion_to_32d("anxious"))
    zero = np.zeros(AFFECT_DIM, dtype=np.float32)
    norms = [float(np.linalg.norm(dyn.step(zero))) for _ in range(5)]
    assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1))


def test_strong_then_zero_monotonic_decay():
    dyn = AffectDynamics(decay=0.8, gain=1.0)
    dyn.step(emotion_to_32d("terrified"))
    zero = np.zeros(AFFECT_DIM, dtype=np.float32)
    norms = [float(np.linalg.norm(dyn.step(zero))) for _ in range(4)]
    assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1))


def test_dynamics_reset_clears_state():
    dyn = AffectDynamics()
    dyn.step(emotion_to_32d("joyful"))
    dyn.reset()
    assert dyn.state is None
    assert dyn.trajectory == []
