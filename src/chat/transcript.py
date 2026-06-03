"""Format chat history for Llama instruct template and TRIBEv2 events."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from src.chat.session import ChatMessage


def messages_to_chat_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def build_llama_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[ChatMessage],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Apply Llama 3.2 instruct chat template."""
    chat = messages_to_chat_dicts(messages)
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def format_tribev2_transcript(
    messages: list[ChatMessage],
    *,
    session_start: float | None = None,
) -> str:
    """
    Timeline text for TRIBEv2 text stimulus.

    Lines: [MM:SS] role: content
    """
    if not messages:
        return ""
    t0 = session_start if session_start is not None else messages[0].timestamp
    lines: list[str] = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        elapsed = max(0.0, m.timestamp - t0)
        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {m.role}: {m.content}")
    return "\n".join(lines)


def write_tribev2_text_file(messages: list[ChatMessage], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = format_tribev2_transcript(messages)
    path.write_text(text, encoding="utf-8")
    return path


def trim_messages_by_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[ChatMessage],
    max_tokens: int,
) -> list[ChatMessage]:
    """Keep system + most recent turns within max_tokens (approximate)."""
    if not messages:
        return messages
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    kept: list[ChatMessage] = list(system)
    for i in range(len(rest) - 1, -1, -1):
        candidate = system + rest[i:]
        prompt = build_llama_prompt(tokenizer, candidate, add_generation_prompt=True)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n <= max_tokens:
            return candidate
    return system + rest[-2:] if len(rest) >= 2 else system + rest
