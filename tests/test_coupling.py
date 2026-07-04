"""Track B — coupling tests."""

import numpy as np

from src.affective.coupling import affect_coupling_corr, couple
from src.affective.emotion_lexicon import emotion_to_32d


def test_couple_moves_toward_user():
    internal = emotion_to_32d("neutral")
    user = emotion_to_32d("anxious")
    out = couple(user, internal, coupling=0.5)
    assert np.linalg.norm(out - user) < np.linalg.norm(internal - user)


def test_coupling_corr_identical_trajectories():
    t = [emotion_to_32d("sad"), emotion_to_32d("anxious"), emotion_to_32d("afraid")]
    r = affect_coupling_corr(t, t)
    assert r >= 0.99


def test_couple_monotonic_in_coupling_strength():
    internal = emotion_to_32d("neutral")
    user = emotion_to_32d("anxious")
    dists = [
        float(np.linalg.norm(couple(user, internal, coupling=c) - user))
        for c in (0.1, 0.5, 0.9)
    ]
    assert dists[0] > dists[1] > dists[2]


def test_couple_neutral_user_moves_less_than_anxious():
    internal = emotion_to_32d("grateful")
    neutral_user = emotion_to_32d("neutral")
    anxious_user = emotion_to_32d("anxious")
    out_neutral = couple(neutral_user, internal, coupling=0.4)
    out_anxious = couple(anxious_user, internal, coupling=0.4)
    assert np.linalg.norm(out_neutral - internal) < np.linalg.norm(out_anxious - internal)


def test_couple_none_internal_returns_user():
    user = emotion_to_32d("grateful")
    out = couple(user, None, coupling=0.5)
    assert np.allclose(out, user)
