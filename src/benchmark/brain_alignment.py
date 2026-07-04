"""Encoder vs measured ROI alignment (Track F v2 — optional)."""

from __future__ import annotations

from typing import Any

import numpy as np


def alignment_report(
    *,
    encoder_vectors: np.ndarray | None = None,
    roi_vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Pearson r between encoder VAD and ROI-derived valence/arousal.

    Returns scientific=False when fMRI preprocessing (OpenNeuro ds002322) is absent.
    """
    if encoder_vectors is None or roi_vectors is None:
        return {
            "scientific": False,
            "r": 0.0,
            "reason": "no_fMRI_data",
            "metric_type": "alignment",
        }
    enc = np.asarray(encoder_vectors, dtype=np.float64)
    roi = np.asarray(roi_vectors, dtype=np.float64)
    if enc.shape != roi.shape or enc.size < 3:
        return {
            "scientific": False,
            "r": 0.0,
            "reason": "shape_mismatch",
            "metric_type": "alignment",
        }
    if enc.std() < 1e-8 or roi.std() < 1e-8:
        return {
            "scientific": False,
            "r": 0.0,
            "reason": "zero_variance",
            "metric_type": "alignment",
        }
    r = float(np.corrcoef(enc.ravel(), roi.ravel())[0, 1])
    return {
        "scientific": False,
        "r": round(r, 4),
        "reason": "v2_holdout_not_run",
        "metric_type": "alignment",
    }
