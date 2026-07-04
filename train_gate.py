"""
Train AffectiveGate on frozen Llama with encoder-derived affect vectors (M1 Track I).

Usage:
  modal run train_gate.py
  modal run train_gate.py --max-samples 50 --epochs 1
  py -3 train_gate.py --local --fixture   # requires CUDA; CI uses tests only
"""

from __future__ import annotations

import json

from src.common import app
from src.config import GATE_TRAIN_EPOCHS, GATE_TRAIN_MAX_SAMPLES
from src.runtime_paths import empatheticdialogues_dir, gate_ckpt_dir
from src.train import modal_jobs as _modal_jobs  # noqa: F401
from src.train.gate import train_gate_loop
from src.train.modal_jobs import train_gate_remote


@app.local_entrypoint(name="train_gate")
def main(max_samples: int = GATE_TRAIN_MAX_SAMPLES, epochs: int = GATE_TRAIN_EPOCHS):
    print(json.dumps(train_gate_remote.remote(max_samples=max_samples, epochs=epochs), indent=2))


def train_local(*, fixture: bool = False, max_samples: int = 20, epochs: int = 1) -> dict:
    from src.config import PROJECT_ROOT

    data_dir = (
        PROJECT_ROOT / "tests" / "fixtures" / "empatheticdialogues"
        if fixture
        else empatheticdialogues_dir()
    )
    return train_gate_loop(
        data_dir=data_dir,
        out_dir=gate_ckpt_dir(),
        max_samples=max_samples,
        epochs=epochs,
        backend="hash" if fixture else None,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    if args.local:
        print(json.dumps(train_local(fixture=args.fixture, max_samples=args.max_samples, epochs=args.epochs), indent=2))
