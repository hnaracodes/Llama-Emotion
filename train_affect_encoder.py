"""
Train AffectEncoder on EmpatheticDialogues label reconstruction (§1 M0/M1).

Usage:
  modal run train_affect_encoder.py
  modal run train_affect_encoder.py --epochs 3
  py -3 train_affect_encoder.py --local --fixture   # offline CI only
"""

from __future__ import annotations

import json

from src.common import app
from src.config import (
    AFFECT_ENCODER_BATCH_SIZE,
    AFFECT_ENCODER_TRAIN_EPOCHS,
    AFFECT_ENCODER_TRAIN_LR,
)
from src.runtime_paths import affect_encoder_dir, empatheticdialogues_dir
from src.train import modal_jobs as _modal_jobs  # noqa: F401 — register Modal functions
from src.train.encoder import train_encoder_loop
from src.train.modal_jobs import train_encoder_remote


@app.local_entrypoint(name="train_affect_encoder")
def main(
    epochs: int = AFFECT_ENCODER_TRAIN_EPOCHS,
    lr: float = AFFECT_ENCODER_TRAIN_LR,
    batch_size: int = AFFECT_ENCODER_BATCH_SIZE,
):
    print(json.dumps(train_encoder_remote.remote(epochs=epochs, lr=lr, batch_size=batch_size), indent=2))


def train_local(
    *,
    fixture: bool = False,
    epochs: int = 2,
    max_samples: int = 32,
    backend: str | None = None,
) -> dict:
    from src.config import PROJECT_ROOT

    data_dir = (
        PROJECT_ROOT / "tests" / "fixtures" / "empatheticdialogues"
        if fixture
        else empatheticdialogues_dir()
    )
    return train_encoder_loop(
        data_dir=data_dir,
        out_dir=affect_encoder_dir(),
        epochs=epochs,
        lr=AFFECT_ENCODER_TRAIN_LR,
        batch_size=min(AFFECT_ENCODER_BATCH_SIZE, max_samples),
        max_samples=max_samples,
        backend=backend,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--fixture", action="store_true", help="Train on tests/fixtures mini CSV")
    p.add_argument("--backend", choices=("hybrid", "hash"), default=None)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    if args.local:
        backend = args.backend or ("hash" if args.fixture else None)
        print(json.dumps(train_local(fixture=args.fixture, epochs=args.epochs, backend=backend), indent=2))
