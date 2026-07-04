"""EmpatheticDialogues loader + holdout registry (§1.2–1.4)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from src.affective.emotion_lexicon import emotion_to_32d, normalize_emotion_label
from src.config import (
    BENCHMARK_PROMPT,
    CHAT_AB_TRANSCRIPTS,
    CHAT_AB_USER_QUESTION,
    EMPATHETICDIALOGUES_DIR,
    PHASE4_ABLATION_PROMPTS,
)
from src.runtime_paths import scenario_holdout_dir

SplitName = Literal["train", "valid", "test"]

_SPLIT_FILES: dict[SplitName, tuple[str, ...]] = {
    "train": ("empchat_train.csv", "train.csv"),
    "valid": ("empchat_valid.csv", "valid.csv"),
    "test": ("empchat_test.csv", "test.csv"),
}


@dataclass(frozen=True)
class DialogueSample:
    conv_id: str
    emotion: str
    prompt: str
    utterances: list[str]
    split: SplitName
    target_32d: Any  # np.ndarray — typed as Any to avoid numpy import in frozen dataclass repr

    def transcript_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        for i, utt in enumerate(self.utterances):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": utt})
        return msgs

    def gate_context_and_listener(self) -> tuple[list[dict[str, str]], str] | None:
        """Chat context before last listener turn + listener reply (gate v3 CE target)."""
        if len(self.utterances) < 2:
            return None
        last_listener_idx = len(self.utterances) - 1
        if last_listener_idx % 2 == 0:
            last_listener_idx -= 1
        if last_listener_idx < 1:
            return None
        msgs: list[dict[str, str]] = []
        for i in range(last_listener_idx):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": self.utterances[i]})
        return msgs, self.utterances[last_listener_idx]

    @property
    def listener_reply(self) -> str | None:
        pair = self.gate_context_and_listener()
        return pair[1] if pair else None


def _resolve_csv(data_dir: Path, split: SplitName) -> Path:
    for name in _SPLIT_FILES[split]:
        path = data_dir / name
        if path.is_file():
            return path
    names = ", ".join(_SPLIT_FILES[split])
    raise FileNotFoundError(
        f"No EmpatheticDialogues CSV for split={split!r} under {data_dir}. "
        f"Expected one of: {names}. Run scripts/download_empatheticdialogues.py"
    )


def collect_holdout_texts() -> set[str]:
    """Exact strings that must never appear in training (§1.4 holdout)."""
    texts: set[str] = set()
    texts.add(BENCHMARK_PROMPT.strip())
    for entry in PHASE4_ABLATION_PROMPTS:
        texts.add(entry["prompt"].strip())
    for _scenario, turns in CHAT_AB_TRANSCRIPTS.items():
        for turn in turns:
            texts.add(turn["content"].strip())
    texts.add(CHAT_AB_USER_QUESTION.strip())
    scenario_dir = scenario_holdout_dir()
    if scenario_dir.is_dir():
        from src.benchmark.scenarios import load_holdout_scenarios

        for sc in load_holdout_scenarios(scenario_dir=scenario_dir):
            texts.add(sc.eval_question.strip())
        for path in scenario_dir.glob("*.json"):
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            for turn in data.get("messages", []):
                texts.add(str(turn.get("content", "")).strip())
    return {t for t in texts if t}


class HoldoutRegistry:
    """Reject training samples that leak benchmark holdout prompts."""

    def __init__(self, holdout_texts: set[str] | None = None):
        self.holdout_texts = holdout_texts if holdout_texts is not None else collect_holdout_texts()

    def contains_holdout(self, sample: DialogueSample) -> bool:
        blob = " ".join([sample.prompt] + sample.utterances).strip()
        for h in self.holdout_texts:
            if h and h in blob:
                return True
        return False


def _read_empathetic_csv(path: Path, split: SplitName) -> list[DialogueSample]:
    by_conv: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row.get("conv_id") or row.get("conversation_id") or ""
            if not conv_id:
                continue
            bucket = by_conv.setdefault(
                conv_id,
                {
                    "emotion": normalize_emotion_label(row.get("context", "neutral")),
                    "prompt": row.get("prompt", "").strip(),
                    "turns": [],
                },
            )
            utt = row.get("utterance", "").strip()
            if utt:
                idx = int(row.get("utterance_idx", len(bucket["turns"])))
                bucket["turns"].append((idx, utt))

    samples: list[DialogueSample] = []
    for conv_id, data in by_conv.items():
        turns = [u for _, u in sorted(data["turns"], key=lambda x: x[0])]
        if not turns:
            continue
        emotion = data["emotion"]
        target = emotion_to_32d(emotion)
        samples.append(
            DialogueSample(
                conv_id=str(conv_id),
                emotion=emotion,
                prompt=data["prompt"],
                utterances=turns,
                split=split,
                target_32d=target,
            )
        )
    return samples


class EmpatheticDialoguesDataset:
    def __init__(
        self,
        split: SplitName = "train",
        *,
        data_dir: Path | None = None,
        holdout_registry: HoldoutRegistry | None = None,
        filter_holdouts: bool = True,
    ):
        self.split = split
        self.data_dir = Path(data_dir or EMPATHETICDIALOGUES_DIR)
        self.holdout_registry = holdout_registry or HoldoutRegistry()
        csv_path = _resolve_csv(self.data_dir, split)
        raw = _read_empathetic_csv(csv_path, split)
        if filter_holdouts and split in ("train", "valid"):
            self.samples = [s for s in raw if not self.holdout_registry.contains_holdout(s)]
            self.filtered_count = len(raw) - len(self.samples)
        else:
            self.samples = raw
            self.filtered_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DialogueSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[DialogueSample]:
        yield from self.samples


class ScenarioHoldoutDataset:
    """In-repo benchmark scenarios — eval only."""

    def __init__(self):
        self.samples: list[DialogueSample] = []
        for name, turns in CHAT_AB_TRANSCRIPTS.items():
            utterances = [t["content"] for t in turns]
            emotion = "distress" if name == "distress" else ("hopeful" if name == "hopeful" else "neutral")
            self.samples.append(
                DialogueSample(
                    conv_id=f"chat_ab_{name}",
                    emotion=emotion,
                    prompt=turns[0]["content"],
                    utterances=utterances,
                    split="test",
                    target_32d=emotion_to_32d(emotion),
                )
            )
        for entry in PHASE4_ABLATION_PROMPTS:
            pid = entry["id"]
            prompt = entry["prompt"]
            self.samples.append(
                DialogueSample(
                    conv_id=f"phase4_{pid}",
                    emotion="neutral",
                    prompt=prompt,
                    utterances=[prompt],
                    split="test",
                    target_32d=emotion_to_32d("neutral"),
                )
            )
        scenario_dir = scenario_holdout_dir()
        if scenario_dir.is_dir():
            import json

            for path in sorted(scenario_dir.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                utterances = [
                    str(t.get("content", "")).strip()
                    for t in data.get("messages", [])
                    if str(t.get("content", "")).strip()
                ]
                if not utterances:
                    continue
                emotion = str(data.get("emotion") or data.get("emotion_hint") or "neutral")
                self.samples.append(
                    DialogueSample(
                        conv_id=f"scenario_{path.stem}",
                        emotion=emotion,
                        prompt=utterances[0],
                        utterances=utterances,
                        split="test",
                        target_32d=emotion_to_32d(emotion),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DialogueSample:
        return self.samples[idx]
