"""Chat session state: messages, affect vector, traits, tone timeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Role = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """Multi-turn session with affective state and tone history."""

    messages: list[ChatMessage] = field(default_factory=list)
    affect_vector: np.ndarray | None = None
    traits: dict[str, float] = field(default_factory=dict)
    dominant_tone: str = "neutral"
    last_refresh_ts: float = 0.0
    last_user_ts: float = field(default_factory=time.time)
    tone_timeline: list[dict[str, Any]] = field(default_factory=list)
    hook_strength: float = 1.0
    manual_affect_scale: float | None = None
    affect_dynamics: Any | None = None
    snn_mem_state: Any | None = None
    turn_index: int = 0
    affect_trajectory: list[list[float]] = field(default_factory=list)

    def append(self, role: Role, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        if role == "user":
            self.last_user_ts = time.time()
            self.turn_index += 1

    def reset_affect_state(self) -> None:
        self.affect_vector = None
        self.snn_mem_state = None
        self.affect_dynamics = None
        self.turn_index = 0
        self.affect_trajectory.clear()

    def transcript_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.role in ("user", "assistant")]

    def needs_affect_refresh(self, interval_sec: float) -> bool:
        if not self.transcript_messages():
            return False
        return (time.time() - self.last_refresh_ts) >= interval_sec

    def record_tone_event(
        self,
        *,
        event: str,
        before: str,
        after: str,
        shift: float,
        traits: dict[str, float],
    ) -> None:
        self.tone_timeline.append(
            {
                "ts": time.time(),
                "event": event,
                "before": before,
                "after": after,
                "shift": shift,
                "traits": dict(traits),
            }
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
            "traits": self.traits,
            "dominant_tone": self.dominant_tone,
            "tone_timeline": self.tone_timeline,
            "affect_vector": (
                self.affect_vector.tolist() if self.affect_vector is not None else None
            ),
        }
