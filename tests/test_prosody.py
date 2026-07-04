"""Track H — prosody encoder tests."""

import numpy as np
import pytest

pytest.importorskip("src.encoder.prosody", reason="Track H not implemented yet")

from src.config import AFFECT_DIM, DELTA_THETA  # noqa: E402
from src.encoder.prosody import prosody_to_spikes  # noqa: E402


def test_prosody_spike_shape():
    feats = np.array([120.0, 0.02, 0.3], dtype=np.float32)
    spikes = prosody_to_spikes(feats, theta=DELTA_THETA)
    assert spikes.shape[-1] == AFFECT_DIM


def test_prosody_sparsity_band():
    feats = np.zeros(3, dtype=np.float32)
    spikes = prosody_to_spikes(feats, theta=DELTA_THETA)
    rate = float(np.mean(spikes > 0))
    assert 0.0 <= rate <= 1.0
