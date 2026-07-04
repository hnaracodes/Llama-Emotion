"""Track F — brain alignment metric tests."""

from src.benchmark.brain_alignment import alignment_report


def test_alignment_bounded_without_fmri():
    report = alignment_report(encoder_vectors=None, roi_vectors=None)
    assert report["scientific"] is False
    assert -1.0 <= report["r"] <= 1.0
