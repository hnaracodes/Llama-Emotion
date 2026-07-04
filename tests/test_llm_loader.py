"""generate_text: full-sequence vs newly-generated-only text.

Regression coverage for a bug where collapse detection (and any other
consumer that only cares about what the *model* produced) was handed the
full decoded sequence — prompt + chat template + scripted history plus the
generation — because `tokenizer.decode(out[0], ...)` decodes everything.
`generate_text` must also expose the newly generated continuation alone via
`stats["new_text"]` so callers don't have to re-derive it (or, worse, run
downstream heuristics against benchmark-authored prompt text).
"""

from __future__ import annotations

import torch

from src.llm.loader import generate_text

_PROMPT_LEN = 5
_NEW_LEN = 3


class _FakeTokenizerOutput(dict):
    def to(self, device):  # noqa: ARG002 - device unused for the fake CPU tensors
        return self


class _FakeTokenizer:
    """Maps token ids 1:1 to space-joined words; decode never re-adds a BOS."""

    pad_token_id = 0

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        ids = torch.arange(1, _PROMPT_LEN + 1).unsqueeze(0)
        return _FakeTokenizerOutput(input_ids=ids)

    def decode(self, token_ids, skip_special_tokens: bool = True):  # noqa: ARG002
        return " ".join(f"tok{int(t)}" for t in token_ids.tolist())


class _FakeModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def generate(self, input_ids=None, **_kwargs):
        new_ids = torch.arange(
            _PROMPT_LEN + 1, _PROMPT_LEN + 1 + _NEW_LEN
        ).unsqueeze(0)
        return torch.cat([input_ids, new_ids], dim=1)


def test_generate_text_new_text_excludes_prompt():
    text, stats = generate_text(_FakeModel(), _FakeTokenizer(), "irrelevant prompt")

    full_expected = " ".join(f"tok{i}" for i in range(1, _PROMPT_LEN + _NEW_LEN + 1))
    new_expected = " ".join(
        f"tok{i}" for i in range(_PROMPT_LEN + 1, _PROMPT_LEN + _NEW_LEN + 1)
    )

    assert text == full_expected
    assert stats["new_text"] == new_expected
    assert stats["new_tokens"] == _NEW_LEN
    # The prompt-only tokens must not leak into the generated-only text.
    prompt_only_tokens = {f"tok{i}" for i in range(1, _PROMPT_LEN + 1)}
    assert prompt_only_tokens.isdisjoint(stats["new_text"].split())
