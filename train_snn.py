"""
Phase 3/5: SNN forward pass + optional STDP training on Modal.

Usage:
  modal run train_snn.py
  modal run train_snn.py --stub
  modal run train_snn.py --stdp-steps 50
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np
import torch

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import ARTIFACTS_MOUNT, DELTA_THETA


@app.function(image=affective_image, **gpu_kwargs())
def train_snn_remote(stub: bool = True, stdp_steps: int = 0) -> dict:
    from src.affective.tribev2_client import (
        pipeline_to_spikes,
        run_tribev2_predict,
        save_pipeline_artifacts,
        synthetic_fmri_timeseries,
    )
    from src.brain.lif_network import LIFAmygdala, run_amygdala_on_spikes
    from src.brain.stdp import STDPUpdater
    from src.config import AFFECT_DIM

    if stub:
        fmri, source = synthetic_fmri_timeseries(), "synthetic_stub"
    else:
        fmri, source = run_tribev2_predict()

    pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32))

    model = LIFAmygdala(input_dim=pipe["D"], output_dim=AFFECT_DIM)
    aff, stats = run_amygdala_on_spikes(spikes, model)

    stdp_log = []
    if stdp_steps > 0:
        updater = STDPUpdater()
        model.train()
        for step in range(stdp_steps):
            # Use fc1 layer; proxy post spikes from lif1 output via threshold
            with torch.no_grad():
                x = spikes.unsqueeze(0)
                mem = model.lif1.init_leaky()
                pre_spk = []
                post_spk = []
                for t in range(min(32, x.shape[1])):
                    cur = model.fc1(x[:, t, :])
                    spk, mem = model.lif1(cur, mem)
                    pre_spk.append((x[:, t, :] > 0.5).float())
                    post_spk.append(spk.squeeze(0))
                pre = torch.stack(pre_spk, dim=0).squeeze(1)
                post = torch.stack(post_spk, dim=0)
            delta = updater.update_linear(model.fc1, pre, post)
            stdp_log.append({"step": step, "delta_norm": float(delta.norm().item())})

    out_dir = Path(ARTIFACTS_MOUNT) / "snn"
    save_pipeline_artifacts(
        out_dir,
        fmri,
        pipe,
        {"source": source, "stub": stub, "stdp_steps": stdp_steps},
    )
    ckpt = out_dir / "amygdala.pt"
    torch.save(model.state_dict(), ckpt)

    result = {
        "source": source,
        "spike_sparsity": pipe["spike_sparsity"],
        "affective_vector": aff.tolist(),
        "snn_stats": stats,
        "stdp_log": stdp_log[:10],
        "checkpoint": str(ckpt),
    }

    meta_path = Path(ARTIFACTS_MOUNT) / "benchmarks" / "train_snn.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(stub: bool = True, stdp_steps: int = 0):
    result = train_snn_remote.remote(stub=stub, stdp_steps=stdp_steps)
    print(json.dumps(result, indent=2))
