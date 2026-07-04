"""Resolve data and artifact paths for local dev vs Modal GPU containers."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import (
    AFFECT_ENCODER_DIR,
    ARTIFACTS_DIR,
    ARTIFACTS_MOUNT,
    EMOTION_LEXICON_PATH,
    EMPATHETICDIALOGUES_DIR,
    GATE_CKPT_DIR,
    PROJECT_ROOT,
    SCENARIO_HOLDOUT_DIR,
    SNN_CKPT_DIR,
)

_MODAL_BAKED_DATA = Path("/opt/saa/data")


def is_modal_runtime() -> bool:
    return os.environ.get("SAA_RUNTIME") == "modal"


def empatheticdialogues_dir() -> Path:
    if is_modal_runtime():
        return Path(ARTIFACTS_MOUNT) / "data" / "raw" / "empatheticdialogues"
    return EMPATHETICDIALOGUES_DIR


def scenario_holdout_dir() -> Path:
    if is_modal_runtime():
        baked = _MODAL_BAKED_DATA / "scenarios"
        if baked.is_dir():
            return baked
        return Path(ARTIFACTS_MOUNT) / "data" / "scenarios"
    return SCENARIO_HOLDOUT_DIR


def emotion_lexicon_path() -> Path:
    if is_modal_runtime():
        baked = _MODAL_BAKED_DATA / "lexicon" / "emotion_lexicon.json"
        if baked.is_file():
            return baked
        return Path(ARTIFACTS_MOUNT) / "data" / "lexicon" / "emotion_lexicon.json"
    return EMOTION_LEXICON_PATH


def affect_encoder_dir() -> Path:
    if is_modal_runtime():
        return Path(ARTIFACTS_MOUNT) / "affect"
    return AFFECT_ENCODER_DIR


def snn_ckpt_dir() -> Path:
    if is_modal_runtime():
        return Path(ARTIFACTS_MOUNT) / "snn"
    return SNN_CKPT_DIR


def gate_ckpt_dir() -> Path:
    if is_modal_runtime():
        return Path(ARTIFACTS_MOUNT) / "gate"
    return GATE_CKPT_DIR


def artifacts_dir() -> Path:
    if is_modal_runtime():
        return Path(ARTIFACTS_MOUNT)
    return ARTIFACTS_DIR


def benchmarks_dir() -> Path:
    return artifacts_dir() / "benchmarks"


def project_root() -> Path:
    if is_modal_runtime():
        return Path("/root")
    return PROJECT_ROOT
