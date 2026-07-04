"""Perception → internal affect coupling (Track B)."""

from __future__ import annotations

import numpy as np

from src.config import AFFECT_DIM


def couple(
    user_vec: np.ndarray,
    internal_state: np.ndarray | None,
    *,
    coupling: float = 0.4,
) -> np.ndarray:
    """
    Drive internal affect toward perceived user affect with inertia from prior state.
    """
    user_vec = np.asarray(user_vec, dtype=np.float32).reshape(-1)
    if user_vec.shape[0] != AFFECT_DIM:
        raise ValueError(f"user_vec dim {user_vec.shape[0]} != {AFFECT_DIM}")
    if internal_state is None:
        return user_vec.copy()
    internal_state = np.asarray(internal_state, dtype=np.float32).reshape(-1)
    return ((1.0 - coupling) * internal_state + coupling * user_vec).astype(np.float32)


def affect_coupling_corr(user_traj: list[np.ndarray], internal_traj: list[np.ndarray]) -> float:
    """Pearson r between consecutive deltas (Track B metric)."""
    if len(user_traj) < 2 or len(internal_traj) < 2:
        return 0.0
    u = np.array([np.linalg.norm(user_traj[i] - user_traj[i - 1]) for i in range(1, len(user_traj))])
    v = np.array(
        [np.linalg.norm(internal_traj[i] - internal_traj[i - 1]) for i in range(1, len(internal_traj))]
    )
    if u.std() < 1e-8 or v.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])
