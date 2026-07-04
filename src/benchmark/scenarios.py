"""Load holdout multi-turn scenarios from data/scenarios/ (eval only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from src.runtime_paths import scenario_holdout_dir

ScenarioCategory = Literal[
    "distress_recovery",
    "conflict",
    "factual_neutral",
    "tone_shift",
]

VALID_CATEGORIES: frozenset[str] = frozenset(
    {"distress_recovery", "conflict", "factual_neutral", "tone_shift"}
)


@dataclass(frozen=True)
class HoldoutScenario:
    id: str
    category: ScenarioCategory
    tags: tuple[str, ...]
    messages: tuple[dict[str, str], ...]
    emotion_hint: str
    eval_question: str

    @property
    def turn_count(self) -> int:
        return len(self.messages)

    def transcript_messages(self) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


def _parse_scenario(data: dict[str, Any], *, path_stem: str) -> HoldoutScenario | None:
    msgs = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", "")).strip()}
        for m in data.get("messages", [])
        if str(m.get("content", "")).strip()
    ]
    if not msgs:
        return None
    sid = str(data.get("id") or path_stem)
    raw_cat = str(data.get("category") or "distress_recovery")
    category: ScenarioCategory = (
        raw_cat if raw_cat in VALID_CATEGORIES else "distress_recovery"  # type: ignore[assignment]
    )
    tags = tuple(str(t) for t in data.get("tags", ()))
    emotion = str(data.get("emotion") or data.get("emotion_hint") or "neutral")
    question = str(
        data.get("eval_question")
        or "What should I do next?"
    )
    return HoldoutScenario(
        id=sid,
        category=category,
        tags=tags,
        messages=tuple(msgs),
        emotion_hint=emotion,
        eval_question=question,
    )


def load_holdout_scenarios(
    *,
    scenario_dir: Path | None = None,
    category: ScenarioCategory | None = None,
    min_turns: int = 0,
    tag: str | None = None,
) -> list[HoldoutScenario]:
    """Load sorted holdout scenarios; optional filters."""
    root = scenario_dir or scenario_holdout_dir()
    out: list[HoldoutScenario] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sc = _parse_scenario(data, path_stem=path.stem)
        if sc is None:
            continue
        if category is not None and sc.category != category:
            continue
        if sc.turn_count < min_turns:
            continue
        if tag is not None and tag not in sc.tags:
            continue
        out.append(sc)
    return out


def iter_scenario_messages(scenario: HoldoutScenario) -> Iterator[dict[str, str]]:
    yield from scenario.transcript_messages()
