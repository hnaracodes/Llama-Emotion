"""Gate affect vector builder tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.brain.lif_network import LIFAmygdala
from src.config import AFFECT_DIM
from src.train.gate_vector import build_gate_affect_vector


@patch("src.train.gate_vector.run_encoder_pipeline")
@patch("src.train.gate_vector.sequence_affective_vectors")
def test_build_gate_affect_vector_clips_norm(mock_seq, mock_pipe):
    pipe = {
        "spikes": np.zeros((4, AFFECT_DIM), dtype=np.float32),
        "affective_vectors": np.ones((4, AFFECT_DIM), dtype=np.float32) * 5.0,
    }
    mock_pipe.return_value = (pipe, "encoder:test")
    aff = np.ones(AFFECT_DIM, dtype=np.float32) * 5.0
    mock_seq.return_value = (np.stack([aff] * 4), ("m1", "m2"), {})

    encoder = MagicMock()
    amy = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    vec, mem, out_pipe = build_gate_affect_vector(
        [{"role": "user", "content": "hello"}],
        encoder=encoder,
        amygdala=amy,
        device="cpu",
    )
    assert out_pipe is pipe
    assert mem == ("m1", "m2")
    assert vec.shape == (AFFECT_DIM,)
    assert float(torch.linalg.norm(vec).item()) <= 1.0 + 1e-5
