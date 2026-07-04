"""Phase loop coupling logic tests."""

import numpy as np

from src.affective.coupling import affect_coupling_corr, couple
from src.affective.dynamics import AffectDynamics
from src.affective.emotion_lexicon import emotion_to_32d


def test_loop_coupling_threads_prior_state():
    """N-2 fix: couple(user, prior_state) not couple(user, amygdala_t)."""
    user_vecs = [emotion_to_32d(e) for e in ("sad", "anxious", "afraid")]
    dyn = AffectDynamics()
    internal = []
    state = None
    for u in user_vecs:
        state = couple(u, state, coupling=0.35)
        internal.append(dyn.step(state))
    assert len(internal) == 3
    r = affect_coupling_corr(user_vecs, internal)
    assert -1.0 <= r <= 1.0
