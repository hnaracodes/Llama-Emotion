"""Tests for AffectEncoder."""

import numpy as np
import torch

from src.affective.encoder import AffectEncoder, lexicon_baseline_vector
from src.config import AFFECT_DIM


def test_encode_text_shape():
    enc = AffectEncoder(output_dim=AFFECT_DIM, backend="hash")
    vec = enc.encode_text("I feel anxious about tomorrow.")
    assert vec.shape == (AFFECT_DIM,)


def test_encode_transcript_timesteps():
    enc = AffectEncoder(backend="hash")
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    traj = enc.encode_transcript(msgs)
    assert traj.shape == (2, AFFECT_DIM)


def test_checkpoint_roundtrip(tmp_path):
    enc = AffectEncoder(backend="hash")
    path = tmp_path / "encoder.pt"
    enc.save_checkpoint(path, extra={"valid_vad_mse": 0.1})
    loaded, meta = AffectEncoder.load_checkpoint(path)
    assert meta["supervision"] == "empatheticdialogues_v1"
    enc.eval()
    loaded.eval()
    t = "test utterance"
    np.testing.assert_allclose(enc.encode_text(t), loaded.encode_text(t), rtol=1e-5)


def test_lexicon_baseline_differs_by_emotion():
    a = lexicon_baseline_vector("anxious")
    b = lexicon_baseline_vector("grateful")
    assert not np.allclose(a[:3], b[:3])
