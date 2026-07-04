"""Checkpoint load/save for encoder, SNN amygdala, and affective gate (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import (
    AFFECT_DIM,
    AFFECT_ENCODER_CKPT_NAME,
    AMYGDALA_CKPT_NAME,
    GATE_CKPT_NAME,
)
from src.runtime_paths import (
    affect_encoder_dir,
    gate_ckpt_dir,
    snn_ckpt_dir,
)


@dataclass
class LoadResult:
    source: str  # trained | random_init | missing
    path: str | None = None
    meta: dict[str, Any] | None = None


def _resolve(path: Path | None, default_dir: Path, name: str) -> Path:
    if path is not None:
        return path
    return default_dir / name


def load_encoder(
    path: Path | None = None,
    *,
    device: str = "cpu",
    backend: str | None = None,
) -> tuple[Any, LoadResult]:
    from src.affective.encoder import AffectEncoder

    ckpt = _resolve(path, affect_encoder_dir(), AFFECT_ENCODER_CKPT_NAME)
    if not ckpt.is_file():
        enc = AffectEncoder(backend=backend or "hash")
        return enc, LoadResult(source="random_init", path=None)
    model, meta = AffectEncoder.load_checkpoint(ckpt, map_location=device)
    source = "trained" if meta.get("supervision") else "unverified_checkpoint"
    return model, LoadResult(source=source, path=str(ckpt), meta=meta)


def load_amygdala(
    path: Path | None = None,
    *,
    input_dim: int = AFFECT_DIM,
    device: str = "cpu",
) -> tuple[nn.Module, LoadResult]:
    from src.brain.lif_network import LIFAmygdala

    ckpt = _resolve(path, snn_ckpt_dir(), AMYGDALA_CKPT_NAME)
    model = LIFAmygdala(input_dim=input_dim, output_dim=AFFECT_DIM)
    if not ckpt.is_file():
        model.to(device)
        return model, LoadResult(source="random_init", path=None)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    extra: dict[str, Any] = {}
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
        extra = {k: v for k, v in state.items() if k != "state_dict"}
    else:
        model.load_state_dict(state)
    model.to(device)
    source = "trained" if extra.get("supervision") else "unverified_checkpoint"
    return model, LoadResult(
        source=source,
        path=str(ckpt),
        meta=extra or None,
    )


def load_gate(
    gate: nn.Module,
    *,
    path: Path | None = None,
    model_id: str,
    hidden_size: int,
    device: str = "cpu",
) -> LoadResult:
    ckpt = _resolve(path, gate_ckpt_dir(), GATE_CKPT_NAME)
    if not ckpt.is_file():
        gate.to(device)
        return LoadResult(source="random_init", path=None)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if payload.get("model_id") and payload["model_id"] != model_id:
        raise ValueError(
            f"Gate checkpoint model_id={payload['model_id']!r} != {model_id!r}"
        )
    if int(payload.get("hidden_size", hidden_size)) != hidden_size:
        raise ValueError(
            f"Gate hidden_size mismatch: ckpt={payload.get('hidden_size')} model={hidden_size}"
        )
    gate.load_state_dict(payload["state_dict"])
    gate.to(device)
    assert_gate_noop(gate, device=device)
    return LoadResult(source="trained", path=str(ckpt), meta=payload)


def assert_gate_noop(gate: nn.Module, *, device: str = "cpu", eps: float = 1e-3) -> None:
    """AF-4: trained gate must not inject bias at zero affect vector."""
    gate.eval()
    with torch.no_grad():
        z = torch.zeros(gate.proj.in_features, device=device)
        mod = gate(z)
        norm = float(torch.linalg.norm(mod).item())
    if norm > eps:
        raise ValueError(f"Gate(0) norm {norm:.6f} exceeds eps={eps}")


def save_gate(
    gate: nn.Module,
    path: Path | None = None,
    *,
    model_id: str,
    hidden_size: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    assert_gate_noop(gate, device=str(next(gate.parameters()).device))
    out = _resolve(path, gate_ckpt_dir(), GATE_CKPT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": gate.state_dict(),
        "model_id": model_id,
        "hidden_size": hidden_size,
        "affect_dim": AFFECT_DIM,
        **(extra or {}),
    }
    torch.save(payload, out)
    return out


def save_amygdala(model: nn.Module, path: Path | None = None, *, extra: dict | None = None) -> Path:
    out = _resolve(path, snn_ckpt_dir(), AMYGDALA_CKPT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **(extra or {})}, out)
    return out


def artifact_roots() -> dict[str, Path]:
    from src.runtime_paths import artifacts_dir

    return {
        "encoder": affect_encoder_dir(),
        "snn": snn_ckpt_dir(),
        "gate": gate_ckpt_dir(),
        "artifacts": artifacts_dir(),
    }
