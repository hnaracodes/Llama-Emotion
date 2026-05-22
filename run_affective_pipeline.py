"""Phase 2 Modal entrypoint: TRIBEv2/synthetic → 32-d → spikes."""

from __future__ import annotations

import json

import modal

from src.common import affective_image, app, gpu_kwargs, model_volume
from src.config import ARTIFACTS_MOUNT, DELTA_THETA


@app.function(image=affective_image, **gpu_kwargs())
def run_phase2() -> dict:
    from pathlib import Path

    from src.affective.tribev2_client import (
        pipeline_to_spikes,
        run_tribev2_predict,
        save_pipeline_artifacts,
    )

    fmri, source = run_tribev2_predict()
    pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
    out_dir = Path(ARTIFACTS_MOUNT) / "phase2"
    save_pipeline_artifacts(
        out_dir,
        fmri,
        pipe,
        {"source": source, "theta": DELTA_THETA},
    )
    result = {"source": source, **{k: v for k, v in pipe.items() if k != "affective_vectors"}}
    result["affective_shape"] = list(pipe["affective_vectors"].shape)
    model_volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(run_phase2.remote(), indent=2))
