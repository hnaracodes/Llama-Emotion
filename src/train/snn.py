"""SNN amygdala training loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import AFFECT_DIM, AFFECT_ENCODER_BACKEND, SUPERVISION_VERSION
from src.runtime_paths import benchmarks_dir


def train_snn_loop(
    *,
    data_dir: Path,
    out_dir: Path,
    max_samples: int | None = 500,
    epochs: int = 2,
    lr: float = 1e-3,
    backend: str | None = None,
) -> dict:
    from src.affective.dataset import EmpatheticDialoguesDataset
    from src.affective.pipeline import run_encoder_pipeline
    from src.brain.checkpoints import load_encoder, save_amygdala
    from src.brain.lif_network import LIFAmygdala

    encoder, enc_load = load_encoder(backend=backend or AFFECT_ENCODER_BACKEND)
    train_ds = EmpatheticDialoguesDataset("train", data_dir=data_dir)

    model = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    indices = list(range(len(train_ds)))
    if max_samples:
        indices = indices[:max_samples]

    history = []
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total = 0.0
        n = 0
        model.train()
        for idx in indices:
            sample = train_ds[idx]
            msgs = sample.transcript_messages()
            pipe, _ = run_encoder_pipeline(msgs, encoder=encoder)
            target = torch.from_numpy(np.asarray(sample.target_32d, dtype=np.float32))
            spikes = torch.from_numpy(pipe["spikes"].astype(np.float32))
            if spikes.shape[0] < 2:
                pad = torch.zeros(1, spikes.shape[1])
                spikes = torch.cat([pad, spikes], dim=0)
            aff, _ = model(spikes)
            loss = F.mse_loss(aff, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        history.append({"epoch": epoch, "train_mse": total / max(n, 1)})

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_amygdala(
        model,
        out_dir / "amygdala.pt",
        extra={"supervision": SUPERVISION_VERSION, "encoder_source": enc_load.source},
    )

    result = {
        "supervision": SUPERVISION_VERSION,
        "tribev2_used": False,
        "encoder_source": enc_load.source,
        "samples": len(indices),
        "checkpoint": str(ckpt),
        "history": history,
    }
    meta = benchmarks_dir() / "train_snn.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
