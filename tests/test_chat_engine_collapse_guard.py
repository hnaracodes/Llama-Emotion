"""Phase 1A/1B (docs/chat_hardening_plan.md): runtime collapse guard and gate
health checks in ChatEngine — no GPU required."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.brain.checkpoints import LoadResult
from src.brain.lif_network import LIFAmygdala
from src.affective.encoder import AffectEncoder
from src.chat.engine import ChatEngine
from src.config import AFFECT_DIM, CHAT_COLLAPSE_FALLBACK_REPLY, GATE_VERSION

_COLLAPSED_TEXT = "hereunderstandhereunderstandhereunderstandhereunderstandhereunderstand"
_CLEAN_TEXT = "That sounds like a genuinely hard week — I'm glad you kept going."


def _mock_model():
    model = MagicMock()
    model.config.hidden_size = 2048
    p = torch.nn.Parameter(torch.zeros(1))
    model.parameters = lambda: iter([p])
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    return model


def _mock_tokenizer(decode_side_effect):
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.decode.side_effect = decode_side_effect
    tok.side_effect = lambda *a, **k: {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    return tok


def _build_engine(model, tok, mock_gate_result: LoadResult):
    enc = AffectEncoder(backend="hash")
    amy = LIFAmygdala(input_dim=AFFECT_DIM, output_dim=AFFECT_DIM)
    with patch("src.brain.checkpoints.load_encoder", return_value=(enc, LoadResult(source="random_init"))), \
         patch("src.brain.checkpoints.load_amygdala", return_value=(amy, LoadResult(source="random_init"))), \
         patch("src.brain.checkpoints.load_gate", return_value=mock_gate_result):
        return ChatEngine(model, tok, encoder_backend="hash")


@patch("src.chat.engine.build_llama_prompt", return_value="prompt")
@patch("src.chat.engine.trim_messages_by_tokens", side_effect=lambda tok, msgs, n: msgs)
@patch("src.chat.engine.register_affective_hooks", return_value=[MagicMock()])
def test_collapse_guard_recovers_with_hooks_off_retry(mock_hooks, _trim, _prompt):
    model = _mock_model()
    tok = _mock_tokenizer([_COLLAPSED_TEXT, _CLEAN_TEXT])
    engine = _build_engine(model, tok, LoadResult(source="trained", meta={"gate_version": GATE_VERSION}))

    with patch.object(
        engine, "refresh_affect", return_value={"ok": True}
    ), patch.object(engine, "_should_modulate", return_value=True):
        result = engine.generate_reply("hello", temperature=0.0)

    assert result["reply"] == _CLEAN_TEXT
    assert result["collapse_detected"] is False
    assert result["recovered"] is True
    # Hooks registered exactly once: the first (collapsed) attempt only —
    # the hooks-off retry must not re-register them.
    assert mock_hooks.call_count == 1
    assert engine.session.turn_metrics[-1]["recovered"] is True
    assert engine.session.turn_metrics[-1]["collapse_detected"] is False


@patch("src.chat.engine.build_llama_prompt", return_value="prompt")
@patch("src.chat.engine.trim_messages_by_tokens", side_effect=lambda tok, msgs, n: msgs)
@patch("src.chat.engine.register_affective_hooks", return_value=[MagicMock()])
def test_collapse_guard_falls_back_when_retry_still_collapses(mock_hooks, _trim, _prompt):
    model = _mock_model()
    tok = _mock_tokenizer([_COLLAPSED_TEXT, _COLLAPSED_TEXT])
    engine = _build_engine(model, tok, LoadResult(source="trained", meta={"gate_version": GATE_VERSION}))

    with patch.object(
        engine, "refresh_affect", return_value={"ok": True}
    ), patch.object(engine, "_should_modulate", return_value=True):
        result = engine.generate_reply("hello", temperature=0.0)

    assert result["reply"] == CHAT_COLLAPSE_FALLBACK_REPLY
    assert result["collapse_detected"] is True
    assert result["recovered"] is False
    assert mock_hooks.call_count == 1


@patch("src.chat.engine.build_llama_prompt", return_value="prompt")
@patch("src.chat.engine.trim_messages_by_tokens", side_effect=lambda tok, msgs, n: msgs)
@patch("src.chat.engine.register_affective_hooks", return_value=[MagicMock()])
def test_collapse_guard_falls_back_immediately_when_hooks_already_off(mock_hooks, _trim, _prompt):
    """No hooks to retry without — collapse must not surface raw looping text."""
    model = _mock_model()
    tok = _mock_tokenizer([_COLLAPSED_TEXT])
    engine = _build_engine(model, tok, LoadResult(source="trained", meta={"gate_version": GATE_VERSION}))

    with patch.object(
        engine, "refresh_affect", return_value={"ok": True}
    ), patch.object(engine, "_should_modulate", return_value=False):
        result = engine.generate_reply("hello", temperature=0.0)

    assert result["reply"] == CHAT_COLLAPSE_FALLBACK_REPLY
    assert result["collapse_detected"] is True
    assert result["recovered"] is False
    mock_hooks.assert_not_called()


@patch("src.chat.engine.build_llama_prompt", return_value="prompt")
@patch("src.chat.engine.trim_messages_by_tokens", side_effect=lambda tok, msgs, n: msgs)
@patch("src.chat.engine.register_affective_hooks", return_value=[])
def test_clean_reply_records_turn_metrics_without_guard(mock_hooks, _trim, _prompt):
    model = _mock_model()
    tok = _mock_tokenizer([_CLEAN_TEXT])
    engine = _build_engine(model, tok, LoadResult(source="trained", meta={"gate_version": GATE_VERSION}))
    engine.session.affect_vector = np.zeros(AFFECT_DIM, dtype=np.float32)
    engine._sync_affect_state()

    with patch.object(engine, "refresh_affect", return_value={"ok": True}):
        result = engine.generate_reply("hello", temperature=0.0)

    assert result["reply"] == _CLEAN_TEXT
    assert result["collapse_detected"] is False
    metrics = engine.session.turn_metrics[-1]
    for key in (
        "turn_index",
        "new_text",
        "collapse_detected",
        "collapse_score",
        "recovered",
        "hooks_active",
        "hook_strength",
        "affect_vector_norm",
        "gate_output_norm",
        "elapsed_sec",
    ):
        assert key in metrics


def test_gate_health_trained_and_matching_version_is_healthy():
    model = _mock_model()
    tok = _mock_tokenizer([_CLEAN_TEXT])
    engine = _build_engine(
        model, tok, LoadResult(source="trained", meta={"gate_version": GATE_VERSION})
    )
    health = engine.gate_health()
    assert health["healthy"] is True
    assert health["warning"] is None
    assert health["source"] == "trained"
    assert health["version"] == GATE_VERSION


def test_gate_health_random_init_warns_and_unhealthy():
    model = _mock_model()
    tok = _mock_tokenizer([_CLEAN_TEXT])
    engine = _build_engine(model, tok, LoadResult(source="random_init"))
    health = engine.gate_health()
    assert health["healthy"] is False
    assert "no trained gate" in health["warning"]


def test_gate_health_stale_version_warns_but_source_trained():
    model = _mock_model()
    tok = _mock_tokenizer([_CLEAN_TEXT])
    engine = _build_engine(
        model, tok, LoadResult(source="trained", meta={"gate_version": "v2_contrastive"})
    )
    health = engine.gate_health()
    assert health["source"] == "trained"
    assert health["healthy"] is False
    assert "does not match" in health["warning"]
