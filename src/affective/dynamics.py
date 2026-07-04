"""Leaky affect dynamics for multi-turn state (Track A)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import AFFECT_DECAY, AFFECT_GAIN, AFFECT_DIM


@dataclass
class AffectDynamics:
    decay: float = AFFECT_DECAY
    gain: float = AFFECT_GAIN
    state: np.ndarray | None = None
    trajectory: list[list[float]] = field(default_factory=list)

    def reset(self) -> None:
        self.state = None
        self.trajectory.clear()

    def step(self, new_vec: np.ndarray) -> np.ndarray:
        new_vec = np.asarray(new_vec, dtype=np.float32).reshape(-1)
        if new_vec.shape[0] != AFFECT_DIM:
            raise ValueError(f"Expected affect dim {AFFECT_DIM}, got {new_vec.shape[0]}")
        if self.state is None:
            self.state = new_vec.copy()
        else:
            self.state = (
                self.decay * self.state + self.gain * new_vec
            ).astype(np.float32)
        self.trajectory.append(self.state.tolist())
        return self.state.copy()
