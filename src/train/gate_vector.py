"""Build gate affect vectors on the same encoder→SNN path as chat inference."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from src.affective.affect_norm import clip_affect_norm
from src.affective.pipeline import run_encoder_pipeline
from src.brain.lif_network import LIFAmygdala, MemState, sequence_affective_vectors


def build_gate_affect_vector(
    messages: Sequence[dict[str, str] | Any],
    *,
    encoder,
    amygdala: LIFAmygdala,
    device: torch.device | str,
    snn_mem_state: MemState | None = None,
) -> tuple[torch.Tensor, MemState | None, dict[str, Any]]:
    """
    Encoder pipeline → SNN sliding windows → norm clip.

    Returns (affect_vector, updated_mem_state, encoder_pipe).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    amygdala.eval()

    pipe, _ = run_encoder_pipeline(messages, encoder=encoder)
    spikes = torch.from_numpy(pipe["spikes"].astype(np.float32)).to(device)

    with torch.no_grad():
        aff_seq, new_mem, _ = sequence_affective_vectors(
            spikes, amygdala, mem_state=snn_mem_state
        )

    vec_np = clip_affect_norm(np.asarray(aff_seq[-1], dtype=np.float32))
    vec = torch.from_numpy(vec_np).to(device=device, dtype=torch.float32)
    return vec, new_mem, pipe
