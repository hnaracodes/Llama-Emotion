"""Tests for emotion lexicon → 32-d prototypes."""

import numpy as np

from src.affective.emotion_lexicon import (
    emotion_to_32d,
    emotion_to_vad,
    macro_bucket_scores,
    vad_to_32d,
)
from src.config import AFFECT_DIM


def test_vad_to_32d_shape():
    vec = vad_to_32d(0.5, 0.6, 0.4, emotion="anxious")
    assert vec.shape == (AFFECT_DIM,)
    assert vec[8:].sum() == 0.0


def test_anxious_negative_valence():
    v, a, d = emotion_to_vad("anxious")
    assert v < 0
    assert a > 0.5


def test_distress_bucket_dominates_for_sad():
    scores = macro_bucket_scores("sad")
    assert scores[0] > scores[4]  # distress > neutral


def test_emotion_to_32d_roundtrip_dims():
    vec = emotion_to_32d("grateful")
    assert vec[0] > 0  # valence
    assert 0.0 <= vec[1] <= 1.0
