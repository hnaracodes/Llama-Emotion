"""M1 — engine uses encoder path."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.brain.checkpoints import LoadResult
from src.brain.lif_network import LIFAmygdala
from src.affective.encoder import AffectEncoder
from src.chat.engine import ChatEngine
from src.config import AFFECT_DIM


def _mock_model():
    model = MagicMock()
    model.config.hidden_size = 2048
    p = torch.nn.Parameter(torch.zeros(1))
    model.parameters = lambda: iter([p])
    return model


def _mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    return tok


@patch("src.chat.engine.register_affective_hooks", return_value=[])
@patch("src.brain.checkpoints.load_gate")
@patch("src.brain.checkpoints.load_amygdala")
@patch("src.brain.checkpoints.load_encoder")
def test_refresh_affect_encoder_source(mock_enc, mock_amy, mock_gate, _hooks):
    enc = AffectEncoder(backend="hash")
    amy = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    mock_enc.return_value = (enc, LoadResult(source="random_init"))
    mock_amy.return_value = (amy, LoadResult(source="random_init"))
    mock_gate.return_value = LoadResult(source="random_init")

    engine = ChatEngine(_mock_model(), _mock_tokenizer(), encoder_backend="hash")
    engine.session.append("user", "I feel anxious today.")
    result = engine.refresh_affect(force=True)
    assert result["ok"] is True
    assert result["source"].startswith("encoder:")
    assert engine.session.affect_vector.shape[0] == AFFECT_DIM
    assert engine._hook_handles == []


@patch("src.chat.engine.build_llama_prompt", return_value="prompt")
@patch("src.chat.engine.trim_messages_by_tokens", side_effect=lambda tok, msgs, n: msgs)
@patch("src.chat.engine.register_affective_hooks", return_value=[MagicMock()])
@patch("src.brain.checkpoints.load_gate")
@patch("src.brain.checkpoints.load_amygdala")
@patch("src.brain.checkpoints.load_encoder")
def test_generate_reply_skips_hooks_when_affect_zero(
    mock_enc, mock_amy, mock_gate, mock_hooks, _trim, _prompt
):
    enc = AffectEncoder(backend="hash")
    amy = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    mock_enc.return_value = (enc, LoadResult(source="random_init"))
    mock_amy.return_value = (amy, LoadResult(source="random_init"))
    mock_gate.return_value = LoadResult(source="random_init")

    model = _mock_model()
    tok = _mock_tokenizer()
    tok.decode.return_value = "ok"
    tok.side_effect = lambda *a, **k: {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

    engine = ChatEngine(model, tok, encoder_backend="hash")
    engine.session.affect_vector = np.zeros(AFFECT_DIM, dtype=np.float32)
    engine._sync_affect_state()

    with patch.object(engine, "refresh_affect", return_value={"ok": True}):
        engine.generate_reply("hello", temperature=0.0)
    mock_hooks.assert_not_called()
