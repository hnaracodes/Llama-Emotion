"""Modal GPU jobs for M1 training (importable from src/ on workers)."""

from __future__ import annotations

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import (
    AFFECT_ENCODER_BATCH_SIZE,
    AFFECT_ENCODER_TRAIN_EPOCHS,
    AFFECT_ENCODER_TRAIN_LR,
    GATE_GPU_TIMEOUT_SEC,
    GATE_TRAIN_EPOCHS,
    GATE_TRAIN_MAX_SAMPLES,
)
from src.data.ensure import ensure_empatheticdialogues
from src.runtime_paths import affect_encoder_dir, empatheticdialogues_dir, gate_ckpt_dir, snn_ckpt_dir
from src.train.encoder import train_encoder_loop
from src.train.gate import train_gate_loop
from src.train.snn import train_snn_loop


@app.function(image=affective_image, **gpu_kwargs())
def ensure_data_remote(*, force: bool = False) -> dict:
    path = ensure_empatheticdialogues(empatheticdialogues_dir(), force=force)
    model_volume.commit()
    return {"data_dir": str(path), "force": force}


@app.function(image=affective_image, **gpu_kwargs())
def train_encoder_remote(
    epochs: int = AFFECT_ENCODER_TRAIN_EPOCHS,
    lr: float = AFFECT_ENCODER_TRAIN_LR,
    batch_size: int = AFFECT_ENCODER_BATCH_SIZE,
) -> dict:
    data_dir = ensure_empatheticdialogues(empatheticdialogues_dir())
    result = train_encoder_loop(
        data_dir=data_dir,
        out_dir=affect_encoder_dir(),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    model_volume.commit()
    return result


@app.function(image=affective_image, **gpu_kwargs())
def train_snn_remote(max_samples: int = 500, epochs: int = 2) -> dict:
    data_dir = ensure_empatheticdialogues(empatheticdialogues_dir())
    result = train_snn_loop(
        data_dir=data_dir,
        out_dir=snn_ckpt_dir(),
        max_samples=max_samples,
        epochs=epochs,
    )
    model_volume.commit()
    return result


@app.function(
    image=affective_image,
    **{**gpu_kwargs(), "timeout": GATE_GPU_TIMEOUT_SEC},
)
def train_gate_remote(
    max_samples: int = GATE_TRAIN_MAX_SAMPLES,
    epochs: int = GATE_TRAIN_EPOCHS,
) -> dict:
    data_dir = ensure_empatheticdialogues(empatheticdialogues_dir())
    result = train_gate_loop(
        data_dir=data_dir,
        out_dir=gate_ckpt_dir(),
        max_samples=max_samples,
        epochs=epochs,
    )
    model_volume.commit()
    return result
