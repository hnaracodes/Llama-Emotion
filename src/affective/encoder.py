"""Text → 32-d affect encoder (hybrid: frozen MiniLM + trainable head, or hash fallback)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.affective.emotion_lexicon import emotion_to_32d, normalize_emotion_label
from src.config import AFFECT_DIM, AFFECT_ENCODER_BACKEND, MINILM_MODEL_ID


def _hash_bag(text: str, dim: int = 4096) -> torch.Tensor:
    """Deterministic character n-gram hash bag — offline CI fallback."""
    vec = torch.zeros(dim, dtype=torch.float32)
    t = text.lower().strip()
    if not t:
        return vec
    for n in (3, 4, 5):
        for i in range(max(0, len(t) - n + 1)):
            gram = t[i : i + n]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % dim
            vec[h] += 1.0
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def _make_head(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output_dim),
    )
    nn.init.zeros_(head[-1].bias)
    return head


class AffectEncoder(nn.Module):
    """
    Hybrid encoder (default): frozen MiniLM sentence embedding → trainable MLP → 32-d.

    Targets come from emotion_lexicon (VAD + macro buckets). Hash backend remains for
    fast offline tests (`backend="hash"`).
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        hash_dim: int = 4096,
        hidden: int = 128,
        output_dim: int = AFFECT_DIM,
        minilm_model_id: str = MINILM_MODEL_ID,
        freeze_minilm: bool = True,
    ):
        super().__init__()
        self.backend = (backend or AFFECT_ENCODER_BACKEND).lower()
        self.hash_dim = hash_dim
        self.output_dim = output_dim
        self.minilm_model_id = minilm_model_id
        self.freeze_minilm = freeze_minilm
        self._minilm = None

        if self.backend == "hybrid":
            embed_dim = self._load_minilm().get_embedding_dimension()
            self.head = _make_head(embed_dim, hidden, output_dim)
        elif self.backend == "hash":
            self.head = _make_head(hash_dim, hidden, output_dim)
        else:
            raise ValueError(f"Unknown AffectEncoder backend: {self.backend!r}")

    def _load_minilm(self):
        if self._minilm is None:
            from sentence_transformers import SentenceTransformer

            self._minilm = SentenceTransformer(self.minilm_model_id)
            if self.freeze_minilm:
                for param in self._minilm.parameters():
                    param.requires_grad = False
        return self._minilm

    def _input_features(self, text: str) -> torch.Tensor:
        if self.backend == "hybrid":
            model = self._load_minilm()
            ctx = torch.no_grad() if self.freeze_minilm else torch.enable_grad()
            with ctx:
                emb = model.encode(
                    text or "",
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            # Clone and align device with trainable head (MiniLM may emit CUDA tensors).
            device = next(self.head.parameters()).device
            return emb.float().clone().to(device)
        return _hash_bag(text, self.hash_dim)

    def forward(self, text: str) -> torch.Tensor:
        x = self._input_features(text)
        if x.dim() == 1:
            return self.head(x)
        return self.head(x.unsqueeze(0)).squeeze(0)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        self.eval()
        return self.forward(text).detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_transcript(self, messages: Sequence[Any]) -> np.ndarray:
        self.eval()
        rows: list[np.ndarray] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
            rows.append(self.encode_text(f"{role}: {content}"))
        if not rows:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def encode_training_text(self, sample) -> str:
        if sample.prompt:
            return f"user: {sample.prompt}"
        if sample.utterances:
            return f"user: {sample.utterances[0]}"
        return "user:"

    def save_checkpoint(self, path: Path, *, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "backend": self.backend,
            "hash_dim": self.hash_dim,
            "output_dim": self.output_dim,
            "minilm_model_id": self.minilm_model_id,
            "freeze_minilm": self.freeze_minilm,
            "supervision": "empatheticdialogues_v1",
            **(extra or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, map_location: str = "cpu"
    ) -> tuple[AffectEncoder, dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            backend=payload.get("backend", "hash"),
            hash_dim=int(payload.get("hash_dim", 4096)),
            output_dim=int(payload.get("output_dim", AFFECT_DIM)),
            minilm_model_id=payload.get("minilm_model_id", MINILM_MODEL_ID),
            freeze_minilm=bool(payload.get("freeze_minilm", True)),
        )
        model.load_state_dict(payload["state_dict"])
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
        return model, meta


def lexicon_baseline_vector(emotion: str) -> np.ndarray:
    return emotion_to_32d(normalize_emotion_label(emotion))
