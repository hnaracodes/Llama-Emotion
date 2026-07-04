"""M1 training loops (Modal-safe — lives under src/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AFFECT_DIM,
    AFFECT_ENCODER_BACKEND,
    AFFECT_ENCODER_BATCH_SIZE,
    AFFECT_ENCODER_CKPT_NAME,
    AFFECT_ENCODER_CONTRASTIVE_WEIGHT,
    AFFECT_ENCODER_TRAIN_LR,
    EMOTION_LEXICON_JSON,
    SUPERVISION_VERSION,
)
from src.runtime_paths import affect_encoder_dir, emotion_lexicon_path


def train_encoder_loop(
    *,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    lr: float = AFFECT_ENCODER_TRAIN_LR,
    batch_size: int = AFFECT_ENCODER_BATCH_SIZE,
    max_samples: int | None = None,
    backend: str | None = None,
) -> dict:
    from src.affective.compress import normalize_affective
    from src.affective.dataset import EmpatheticDialoguesDataset
    from src.affective.emotion_lexicon import save_lexicon_json
    from src.affective.encoder import AffectEncoder

    backend = backend or AFFECT_ENCODER_BACKEND
    train_ds = EmpatheticDialoguesDataset("train", data_dir=data_dir)
    valid_ds = EmpatheticDialoguesDataset("valid", data_dir=data_dir)
    test_ds = EmpatheticDialoguesDataset("test", data_dir=data_dir, filter_holdouts=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffectEncoder(output_dim=AFFECT_DIM, backend=backend).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)

    def _vad_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_n = torch.from_numpy(normalize_affective(pred.detach().cpu().numpy().reshape(1, -1))[0])
        tgt_n = torch.from_numpy(normalize_affective(target.detach().cpu().numpy().reshape(1, -1))[0])
        return float(torch.mean(torch.abs(pred_n[:3] - tgt_n[:3])).item())

    def _contrastive_loss(preds: list[torch.Tensor], emotions: list[str]) -> torch.Tensor:
        by_emotion: dict[str, list[torch.Tensor]] = {}
        for pred, emotion in zip(preds, emotions):
            by_emotion.setdefault(emotion, []).append(pred)
        terms: list[torch.Tensor] = []
        for group in by_emotion.values():
            if len(group) < 2:
                continue
            stacked = torch.stack(group)
            centroid = stacked.mean(dim=0, keepdim=True)
            terms.append(F.mse_loss(stacked, centroid.expand_as(stacked)))
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean()

    def run_epoch(split_ds, train: bool) -> tuple[float, float]:
        indices = list(range(len(split_ds)))
        if max_samples is not None:
            indices = indices[:max_samples]
        np.random.shuffle(indices)
        total_mse = 0.0
        total_vad_mae = 0.0
        n = 0
        if train:
            model.train()
        else:
            model.eval()

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_preds: list[torch.Tensor] = []
            batch_targets: list[torch.Tensor] = []
            batch_emotions: list[str] = []

            for idx in batch_idx:
                sample = split_ds[idx]
                text = model.encode_training_text(sample)
                target = torch.from_numpy(np.asarray(sample.target_32d, dtype=np.float32)).to(device)
                if train:
                    pred = model(text)
                    batch_preds.append(pred)
                    batch_targets.append(target)
                    batch_emotions.append(sample.emotion)
                else:
                    with torch.no_grad():
                        pred = model(text)
                    total_mse += float(F.mse_loss(pred, target).item())
                    total_vad_mae += _vad_mae(pred, target)
                    n += 1

            if train and batch_preds:
                mse = torch.stack(
                    [F.mse_loss(p, t) for p, t in zip(batch_preds, batch_targets)]
                ).mean()
                contrastive = _contrastive_loss(batch_preds, batch_emotions)
                loss = mse + AFFECT_ENCODER_CONTRASTIVE_WEIGHT * contrastive
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_mse += float(mse.item())
                for p, t in zip(batch_preds, batch_targets):
                    total_vad_mae += _vad_mae(p.detach(), t)
                n += len(batch_preds)

        return total_mse / max(n, 1), total_vad_mae / max(n, 1)

    history = []
    for epoch in range(epochs):
        tr_mse, _ = run_epoch(train_ds, train=True)
        va_mse, va_vad_mae = run_epoch(valid_ds, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_mse": tr_mse,
                "valid_mse": va_mse,
                "valid_vad_mae": va_vad_mae,
            }
        )

    _, test_vad_mae = run_epoch(test_ds, train=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / AFFECT_ENCODER_CKPT_NAME
    model.save_checkpoint(
        ckpt_path,
        extra={"valid_vad_mae": history[-1]["valid_vad_mae"] if history else None},
    )
    save_lexicon_json(out_dir / EMOTION_LEXICON_JSON)
    save_lexicon_json(emotion_lexicon_path())

    result = {
        "supervision": SUPERVISION_VERSION,
        "encoder_backend": backend,
        "tribev2_used": False,
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
        "filtered_holdout_leaks_train": train_ds.filtered_count,
        "filtered_holdout_leaks_valid": valid_ds.filtered_count,
        "test_vad_mae": test_vad_mae,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    (out_dir / "train_encoder.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
