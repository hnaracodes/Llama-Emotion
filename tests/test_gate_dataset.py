"""Dataset gate training pair extraction."""

from src.affective.dataset import DialogueSample
from src.affective.emotion_lexicon import emotion_to_32d


def test_gate_context_and_listener_two_turn():
    sample = DialogueSample(
        conv_id="0",
        emotion="anxious",
        prompt="I am waiting for exam results.",
        utterances=["I feel sick with worry.", "That sounds really stressful."],
        split="train",
        target_32d=emotion_to_32d("anxious"),
    )
    ctx, reply = sample.gate_context_and_listener()
    assert len(ctx) == 1
    assert ctx[0]["role"] == "user"
    assert reply == "That sounds really stressful."
    assert sample.listener_reply == reply


def test_gate_context_multi_turn_uses_last_listener():
    sample = DialogueSample(
        conv_id="1",
        emotion="sad",
        prompt="prompt",
        utterances=["u1", "a1", "u2", "a2"],
        split="train",
        target_32d=emotion_to_32d("sad"),
    )
    ctx, reply = sample.gate_context_and_listener()
    assert reply == "a2"
    assert len(ctx) == 3
