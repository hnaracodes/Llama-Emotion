"""Compress high-dimensional TRIBEv2 / proxy trajectories to 32-d float32 affective vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class AffectiveCompressor:
    """ROI pooling (v1) or PCA to fixed AFFECT_DIM."""

    def __init__(self, n_components: int = 32, n_roi_groups: int = 32):
        self.n_components = n_components
        self.n_roi_groups = n_roi_groups
        self._pca_components: np.ndarray | None = None
        self._roi_indices: list[np.ndarray] | None = None

    def fit_pca(self, trajectories: np.ndarray) -> "AffectiveCompressor":
        """Fit PCA on (N, V) stacked samples using SVD (no sklearn)."""
        flat = trajectories.reshape(-1, trajectories.shape[-1]).astype(np.float64)
        flat = flat - flat.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        self._pca_components = vt[: self.n_components].astype(np.float32)
        return self

    def fit_roi_groups(self, n_vertices: int) -> "AffectiveCompressor":
        """Split vertices into n_roi_groups contiguous chunks."""
        chunk = max(1, n_vertices // self.n_roi_groups)
        self._roi_indices = []
        for g in range(self.n_roi_groups):
            start = g * chunk
            end = n_vertices if g == self.n_roi_groups - 1 else (g + 1) * chunk
            self._roi_indices.append(np.arange(start, end))
        return self

    def transform_roi(self, ts: np.ndarray) -> np.ndarray:
        """
        ts: (T, V) → (T, n_components) via mean pooling per ROI.
        """
        if self._roi_indices is None:
            self.fit_roi_groups(ts.shape[1])
        out = np.zeros((ts.shape[0], len(self._roi_indices)), dtype=np.float32)
        for i, idx in enumerate(self._roi_indices):
            if len(idx) > 0:
                out[:, i] = ts[:, idx].mean(axis=1)
        return out

    def transform_pca(self, ts: np.ndarray) -> np.ndarray:
        """ts: (T, V) → (T, n_components)."""
        if self._pca_components is None:
            raise RuntimeError("Call fit_pca before transform_pca")
        x = ts.astype(np.float32) - ts.astype(np.float32).mean(axis=0, keepdims=True)
        return (x @ self._pca_components.T).astype(np.float32)

    def transform(self, ts: np.ndarray, method: str = "roi") -> np.ndarray:
        if method == "pca":
            return self.transform_pca(ts)
        return self.transform_roi(ts)

    def save(self, path: Path) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "pca_components": self._pca_components,
                    "roi_indices": self._roi_indices,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "AffectiveCompressor":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_components=data["n_components"])
        obj._pca_components = data.get("pca_components")
        obj._roi_indices = data.get("roi_indices")
        return obj


def normalize_affective(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-timestep L2 normalize (T, D)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    return (vectors / norms).astype(np.float32)
