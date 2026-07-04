"""Optional hybrid encoder test (requires sentence-transformers + network)."""

import pytest

pytest.importorskip("sentence_transformers")

from src.affective.encoder import AffectEncoder
from src.affective.emotion_lexicon import emotion_to_vad
from src.config import AFFECT_DIM


@pytest.mark.slow
def test_hybrid_encoder_produces_finite_output():
    enc = AffectEncoder(backend="hybrid")
    vec = enc.encode_text("I feel anxious about my exam tomorrow.")
    assert vec.shape == (AFFECT_DIM,)
    assert abs(vec).sum() > 0


@pytest.mark.slow
def test_hybrid_checkpoint_roundtrip(tmp_path):
    enc = AffectEncoder(backend="hybrid")
    path = tmp_path / "hybrid_encoder.pt"
    enc.save_checkpoint(path)
    loaded, meta = AffectEncoder.load_checkpoint(path)
    assert meta["backend"] == "hybrid"
    text = "I'm grateful for your help."
    enc.eval()
    loaded.eval()
    import numpy as np

    np.testing.assert_allclose(enc.encode_text(text), loaded.encode_text(text), rtol=1e-5)


def test_lexicon_targets_differ_for_opposite_emotions():
    anxious = emotion_to_vad("anxious")
    grateful = emotion_to_vad("grateful")
    assert anxious[0] < grateful[0]
