"""Tests for encoder-based affect pipeline."""

import numpy as np

from src.affective.encoder import AffectEncoder
from src.affective.pipeline import run_encoder_pipeline
from src.config import AFFECT_DIM


def test_pipeline_shapes_and_source():
    enc = AffectEncoder(backend="hash")
    msgs = [{"role": "user", "content": "I feel overwhelmed."}]
    pipe, source = run_encoder_pipeline(msgs, encoder=enc)
    assert source.startswith("encoder:")
    assert pipe["D"] == AFFECT_DIM
    assert pipe["affective_vectors"].shape[1] == AFFECT_DIM
    assert pipe["spikes"].shape == pipe["affective_vectors"].shape


def test_empty_transcript_returns_neutral_row():
    pipe, source = run_encoder_pipeline([], encoder=AffectEncoder(backend="hash"))
    assert pipe["T"] == 1
    assert np.linalg.norm(pipe["affective_vectors"][-1]) > 0
