"""Unit tests for affective compression."""

import numpy as np

from src.affective.compress import AffectiveCompressor, normalize_affective


def test_roi_compress_shape():
    ts = np.random.randn(20, 256).astype(np.float32)
    comp = AffectiveCompressor(n_components=32)
    comp.fit_roi_groups(256)
    out = comp.transform(ts)
    assert out.shape == (20, 32)


def test_normalize_unit_norm():
    v = np.random.randn(10, 32).astype(np.float32)
    n = normalize_affective(v)
    norms = np.linalg.norm(n, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)
