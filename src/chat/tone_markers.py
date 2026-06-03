"""ANSI tone colors, shift detection, and CLI formatting."""

from __future__ import annotations

import os
import shutil
from typing import Any

from src.config import TONE_FLASH_ON_SHIFT, TONE_SHIFT_THRESHOLD, TONE_USE_COLOR

# tone -> (prefix glyph, fg ANSI, label)
TONE_STYLE: dict[str, tuple[str, str, str]] = {
    "neutral": ("·", "\033[90m", "neutral"),
    "warm": ("~", "\033[33m", "warm"),
    "empathetic": ("≈", "\033[36m", "empathetic"),
    "alert": ("!", "\033[38;5;208m", "alert"),
    "tense": ("▲", "\033[31m", "tense"),
    "calm": ("▼", "\033[32m", "calm"),
    "curious": ("?", "\033[34m", "curious"),
}

RESET = "\033[0m"
BOLD = "\033[1m"
INVERSE = "\033[7m"


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not TONE_USE_COLOR:
        return False
    return True


def dominant_tone(traits: dict[str, float]) -> str:
    """Map trait scalars to a tone bucket (heuristic, display-only)."""
    engagement = traits.get("engagement", 0.0)
    arousal = traits.get("arousal", 0.0)
    tension = traits.get("tension", 0.0)
    warmth = traits.get("warmth", 0.0)
    shift = traits.get("shift", 0.0)
    stability = traits.get("stability", 0.5)

    if shift >= 0.25 and arousal >= 0.5:
        return "alert"
    if tension >= 0.55:
        return "tense"
    if warmth >= 0.55 and engagement >= 0.4 and tension < 0.4:
        return "empathetic"
    if warmth >= 0.5 and engagement >= 0.35:
        return "warm"
    if arousal <= 0.35 and shift <= 0.15 and stability >= 0.6:
        return "calm"
    if engagement >= 0.5 and arousal < 0.5 and tension < 0.45:
        return "curious"
    if engagement < 0.25 and arousal < 0.3:
        return "neutral"
    return "warm" if warmth >= engagement else "neutral"


def tone_prefix(tone: str) -> str:
    glyph, _, _ = TONE_STYLE.get(tone, TONE_STYLE["neutral"])
    return f"[{glyph}]"


def wrap_color(text: str, tone: str) -> str:
    if not color_enabled():
        return text
    _, fg, _ = TONE_STYLE.get(tone, TONE_STYLE["neutral"])
    return f"{fg}{text}{RESET}"


def describe_shift(before: str, after: str) -> str:
    key = (before, after)
    lookup = {
        ("neutral", "warm"): "warming",
        ("warm", "tense"): "more tense",
        ("tense", "calm"): "calming",
        ("calm", "warm"): "warming",
        ("neutral", "empathetic"): "more empathetic",
        ("alert", "calm"): "calming",
        ("neutral", "alert"): "heightened",
    }
    if key in lookup:
        return lookup[key]
    if before == after:
        return "steady"
    return f"{before} → {after}"


def detect_shift(
    before_traits: dict[str, float],
    after_traits: dict[str, float],
    *,
    threshold: float | None = None,
) -> tuple[bool, float, str, str]:
    from src.chat.signatures import trait_shift_magnitude

    threshold = TONE_SHIFT_THRESHOLD if threshold is None else threshold
    magnitude = trait_shift_magnitude(before_traits, after_traits)
    before_tone = dominant_tone(before_traits)
    after_tone = dominant_tone(after_traits)
    shifted = magnitude >= threshold or before_tone != after_tone
    return shifted, magnitude, before_tone, after_tone


def format_shift_banner(
    before: str,
    after: str,
    magnitude: float,
    *,
    minimal: bool | None = None,
) -> str:
    direction = describe_shift(before, after)
    if minimal is None:
        minimal = shutil.get_terminal_size((80, 20)).columns < 60

    core = f"◆ tone: {before} → {after}  (Δ={magnitude:.2f})  [{direction}]"
    if not color_enabled() or not TONE_FLASH_ON_SHIFT:
        return core

    line = f"{BOLD}{INVERSE}{wrap_color(core, after)}{RESET}"
    if minimal:
        return line

    width = min(52, max(len(core) + 4, 40))
    bar = "═" * width
    inner = f"  {core}  ".center(width)
    return f"\n╔{bar}╗\n║{inner}║\n╚{bar}╝"


def format_assistant_line(text: str, tone: str, *, label: str = "Amygdala") -> str:
    prefix = tone_prefix(tone)
    line = f"{prefix} {label}> {text}"
    return wrap_color(line, tone)


def format_affect_line(traits: dict[str, float], source: str = "") -> str:
    parts = [
        f"engagement={traits.get('engagement', 0):.2f}",
        f"arousal={traits.get('arousal', 0):.2f}",
        f"warmth={traits.get('warmth', 0):.2f}",
        f"shift={traits.get('shift', 0):.2f}",
    ]
    suffix = f"  ({source})" if source else ""
    body = "  ".join(parts)
    line = f"[affect] {body}{suffix}"
    tone = dominant_tone(traits)
    return wrap_color(line, tone) if color_enabled() else line


def format_colors_legend() -> str:
    lines = ["Tone → color legend (derived from affective traits, not ground-truth emotion):"]
    for tone, (glyph, _, label) in TONE_STYLE.items():
        sample = wrap_color(f"[{glyph}] {label}", tone)
        lines.append(f"  {sample}")
    lines.append("")
    lines.append(f"Shift banner threshold: Δ ≥ {TONE_SHIFT_THRESHOLD}")
    if os.environ.get("NO_COLOR"):
        lines.append("(NO_COLOR is set — colors disabled)")
    return "\n".join(lines)


def format_mood_summary(
    traits: dict[str, float],
    tone: str,
    *,
    last_refresh_ts: float = 0.0,
) -> str:
    import time

    ago = ""
    if last_refresh_ts > 0:
        ago = f"  (refreshed {int(time.time() - last_refresh_ts)}s ago)"
    lines = [
        f"Current tone: {wrap_color(tone, tone) if color_enabled() else tone}{ago}",
        format_affect_line(traits),
    ]
    return "\n".join(lines)
