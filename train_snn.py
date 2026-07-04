"""
Train SNN amygdala on encoder spike outputs (M1 — §1, replaces TRIBEv2 stub path).

Usage:
  modal run train_snn.py
  modal run train_snn.py --max-samples 500 --epochs 2
  py -3 train_snn.py --local --fixture   # offline CI only
"""

from __future__ import annotations

import json

from src.common import app
from src.runtime_paths import empatheticdialogues_dir, snn_ckpt_dir
from src.train import modal_jobs as _modal_jobs  # noqa: F401
from src.train.modal_jobs import train_snn_remote
from src.train.snn import train_snn_loop


@app.local_entrypoint(name="train_snn")
def main(max_samples: int = 500, epochs: int = 2):
    print(json.dumps(train_snn_remote.remote(max_samples=max_samples, epochs=epochs), indent=2))


def train_local(*, fixture: bool = False, max_samples: int = 100, epochs: int = 2) -> dict:
    from src.config import PROJECT_ROOT

    data_dir = (
        PROJECT_ROOT / "tests" / "fixtures" / "empatheticdialogues"
        if fixture
        else empatheticdialogues_dir()
    )
    return train_snn_loop(
        data_dir=data_dir,
        out_dir=snn_ckpt_dir(),
        max_samples=max_samples,
        epochs=epochs,
        backend="hash" if fixture else None,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    if args.local:
        print(json.dumps(train_local(fixture=args.fixture, max_samples=args.max_samples, epochs=args.epochs), indent=2))
