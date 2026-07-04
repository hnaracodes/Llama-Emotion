"""
M1 training pipeline on Modal GPU (encoder → SNN → gate).

Downloads EmpatheticDialogues to the `saa-models` volume on first run, then trains
all three checkpoints under `/models/{affect,snn,gate}/`.

Usage:
  modal run train_m1.py
  modal run train_m1.py --encoder-epochs 3 --snn-samples 500 --gate-samples 100
  modal run train_m1.py --download-only
"""

from __future__ import annotations

import json

from src.common import app
from src.config import AFFECT_ENCODER_TRAIN_EPOCHS, GATE_TRAIN_MAX_SAMPLES
from src.train import modal_jobs as _modal_jobs  # noqa: F401
from src.train.modal_jobs import (
    ensure_data_remote,
    train_encoder_remote,
    train_gate_remote,
    train_snn_remote,
)


@app.local_entrypoint(name="train_m1")
def main(
    encoder_epochs: int = AFFECT_ENCODER_TRAIN_EPOCHS,
    snn_samples: int = 500,
    snn_epochs: int = 2,
    gate_samples: int = GATE_TRAIN_MAX_SAMPLES,
    gate_epochs: int = 1,
    download_only: bool = False,
    force_download: bool = False,
    skip_gate: bool = False,
):
    data = ensure_data_remote.remote(force=force_download)
    print(json.dumps({"download": data}, indent=2))
    if download_only:
        return

    encoder = train_encoder_remote.remote(epochs=encoder_epochs)
    print(json.dumps({"encoder": encoder}, indent=2))

    snn = train_snn_remote.remote(max_samples=snn_samples, epochs=snn_epochs)
    print(json.dumps({"snn": snn}, indent=2))

    gate = None
    if not skip_gate:
        gate = train_gate_remote.remote(max_samples=gate_samples, epochs=gate_epochs)
        print(json.dumps({"gate": gate}, indent=2))

    summary = {
        "supervision": encoder.get("supervision"),
        "encoder_checkpoint": encoder.get("checkpoint"),
        "snn_checkpoint": snn.get("checkpoint"),
        "gate_checkpoint": gate.get("checkpoint") if gate else None,
        "test_vad_mae": encoder.get("test_vad_mae"),
    }
    print(json.dumps({"m1_summary": summary}, indent=2))
