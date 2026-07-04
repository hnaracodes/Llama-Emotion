"""Map emotion labels → VAD and fixed 32-d affect prototypes (§1 label geometry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AFFECT_DIM

# EmpatheticDialogues 32 situation-emotion labels (+ aliases).
EMOTION_VAD: dict[str, tuple[float, float, float]] = {
    "sentimental": (0.35, 0.45, 0.35),
    "afraid": (-0.65, 0.75, 0.15),
    "proud": (0.70, 0.55, 0.80),
    "faithful": (0.55, 0.35, 0.55),
    "terrified": (-0.85, 0.90, 0.10),
    "joyful": (0.85, 0.70, 0.60),
    "angry": (-0.70, 0.80, 0.55),
    "sad": (-0.75, 0.40, 0.20),
    "jealous": (-0.55, 0.65, 0.35),
    "grateful": (0.75, 0.45, 0.50),
    "prepared": (0.40, 0.40, 0.70),
    "embarrassed": (-0.45, 0.60, 0.15),
    "excited": (0.75, 0.85, 0.55),
    "annoyed": (-0.50, 0.55, 0.45),
    "lonely": (-0.70, 0.35, 0.15),
    "ashamed": (-0.65, 0.55, 0.10),
    "guilty": (-0.60, 0.50, 0.15),
    "surprised": (0.15, 0.80, 0.40),
    "nostalgic": (0.30, 0.40, 0.35),
    "confident": (0.65, 0.50, 0.85),
    "furious": (-0.85, 0.90, 0.65),
    "disappointed": (-0.60, 0.45, 0.25),
    "caring": (0.70, 0.45, 0.55),
    "trusting": (0.60, 0.30, 0.45),
    "disgusted": (-0.70, 0.55, 0.50),
    "anticipating": (0.35, 0.60, 0.45),
    "anxious": (-0.55, 0.75, 0.20),
    "hopeful": (0.65, 0.55, 0.50),
    "content": (0.60, 0.25, 0.45),
    "impressed": (0.55, 0.55, 0.55),
    "appalled": (-0.75, 0.65, 0.40),
    "devastated": (-0.90, 0.50, 0.10),
    # Common aliases / fallbacks
    "neutral": (0.0, 0.25, 0.45),
    "distress": (-0.65, 0.65, 0.20),
    "hopeful_recovery": (0.55, 0.45, 0.45),
}

MACRO_BUCKETS = ("distress", "warmth", "tension", "calm", "neutral")

_BUCKET_MEMBERS: dict[str, set[str]] = {
    "distress": {
        "afraid",
        "terrified",
        "angry",
        "sad",
        "jealous",
        "embarrassed",
        "ashamed",
        "guilty",
        "lonely",
        "anxious",
        "furious",
        "disappointed",
        "appalled",
        "devastated",
        "annoyed",
        "disgusted",
    },
    "warmth": {
        "grateful",
        "caring",
        "trusting",
        "joyful",
        "hopeful",
        "content",
        "impressed",
        "nostalgic",
        "sentimental",
        "proud",
        "faithful",
        "prepared",
        "confident",
        "excited",
        "surprised",
        "anticipating",
    },
    "tension": {
        "angry",
        "furious",
        "terrified",
        "anxious",
        "jealous",
        "annoyed",
        "disgusted",
        "appalled",
        "afraid",
    },
    "calm": {"content", "trusting", "prepared", "faithful", "nostalgic", "sentimental"},
    "neutral": {"neutral"},
}


def normalize_emotion_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def emotion_to_vad(emotion: str) -> tuple[float, float, float]:
    key = normalize_emotion_label(emotion)
    if key in EMOTION_VAD:
        return EMOTION_VAD[key]
    return EMOTION_VAD["neutral"]


def macro_bucket_scores(emotion: str) -> np.ndarray:
    """Soft one-hot-ish scores for 5 macro buckets (dims 3–7 in 32-d layout)."""
    key = normalize_emotion_label(emotion)
    scores = np.zeros(len(MACRO_BUCKETS), dtype=np.float32)
    for i, bucket in enumerate(MACRO_BUCKETS):
        if key in _BUCKET_MEMBERS[bucket]:
            scores[i] = 1.0
    if scores.sum() == 0:
        scores[MACRO_BUCKETS.index("neutral")] = 1.0
    else:
        scores = scores / scores.sum()
    return scores


def vad_to_32d(
    valence: float,
    arousal: float,
    dominance: float,
    *,
    emotion: str | None = None,
) -> np.ndarray:
    """
    Build a 32-d label vector with interpretable prefix (§1.3):
      0–2 VAD, 3–7 macro buckets, 8–31 reserved (zeros for supervision targets).
    """
    vec = np.zeros(AFFECT_DIM, dtype=np.float32)
    vec[0] = float(np.clip(valence, -1.0, 1.0))
    vec[1] = float(np.clip(arousal, 0.0, 1.0))
    vec[2] = float(np.clip(dominance, 0.0, 1.0))
    if emotion is not None:
        vec[3:8] = macro_bucket_scores(emotion)
    return vec


def emotion_to_32d(emotion: str) -> np.ndarray:
    v, a, d = emotion_to_vad(emotion)
    return vad_to_32d(v, a, d, emotion=emotion)


def lexicon_metadata() -> dict[str, Any]:
    return {
        "version": "empatheticdialogues_v1",
        "affect_dim": AFFECT_DIM,
        "vad_dims": [0, 1, 2],
        "macro_bucket_dims": [3, 4, 5, 6, 7],
        "learned_dims": list(range(8, AFFECT_DIM)),
        "emotions": sorted(k for k in EMOTION_VAD if k not in ("neutral", "distress", "hopeful_recovery")),
        "macro_buckets": list(MACRO_BUCKETS),
    }


def save_lexicon_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **lexicon_metadata(),
        "emotion_vad": {k: list(v) for k, v in EMOTION_VAD.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_lexicon_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
