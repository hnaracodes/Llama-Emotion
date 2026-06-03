#!/usr/bin/env python3
"""
Emotional CLI chat — talk to W4 Llama with TRIBEv2-driven affective modulation.

Usage (project .venv only):
  .venv\\Scripts\\python.exe chat.py --local          # requires CUDA
  .venv\\Scripts\\python.exe chat.py --modal          # Modal warm GPU worker

Commands during chat:
  /mood /colors /refresh /strength N /affect high|low|neutral /save /quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.chat.tone_markers import (
    format_affect_line,
    format_assistant_line,
    format_colors_legend,
    format_mood_summary,
    format_shift_banner,
)
from src.config import AFFECT_REFRESH_SEC, CHAT_ASSISTANT_LABEL, CHAT_HOOK_STRENGTH


def _print_shift_if_needed(result: dict) -> None:
    if result.get("shifted"):
        print(
            format_shift_banner(
                result.get("before_tone", "neutral"),
                result.get("after_tone", "neutral"),
                float(result.get("shift_magnitude", 0.0)),
            )
        )


def run_local_cli(args: argparse.Namespace) -> int:
    import torch

    from src.chat.engine import ChatEngine
    from src.llm.loader import load_quantized_llama

    if not torch.cuda.is_available():
        print("No local CUDA — use: .venv\\Scripts\\python.exe chat.py --modal")
        return 1

    print("Loading W4 Llama (local GPU)...")
    model, tokenizer = load_quantized_llama()
    engine = ChatEngine(model, tokenizer, hook_strength=args.strength)
    return _repl(engine, modal_worker=None, args=args)


def run_modal_cli(args: argparse.Namespace) -> int:
    from run_chat import EmotionalChatWorker

    print("Connecting to Modal EmotionalChatWorker (warm GPU)...")
    worker = EmotionalChatWorker()
    return _repl(engine=None, modal_worker=worker, args=args)


def _repl(engine, modal_worker, args: argparse.Namespace) -> int:
    print(f"Emotional CLI Chat — refresh every {AFFECT_REFRESH_SEC}s | /colors for legend")
    print("Type a message or /quit\n")

    last_refresh_check = time.time()

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue

        if user_text.startswith("/"):
            if _handle_command(user_text, engine, modal_worker, args):
                break
            continue

        # Auto refresh check (wall clock)
        if time.time() - last_refresh_check >= AFFECT_REFRESH_SEC:
            _do_refresh(engine, modal_worker)
            last_refresh_check = time.time()

        if modal_worker is not None:
            result = modal_worker.chat_turn.remote(user_text, temperature=args.temperature)
        else:
            result = engine.generate_reply(user_text, temperature=args.temperature)

        tone = result.get("dominant_tone", "neutral")
        print(format_assistant_line(result["reply"], tone, label=CHAT_ASSISTANT_LABEL))
        print(
            format_affect_line(
                result.get("traits", {}),
                source=result.get("tribe_source", ""),
            )
        )
        print()

    _save_if_requested(engine, modal_worker, args)
    if engine is not None:
        engine.cleanup()
    return 0


def _handle_command(cmd: str, engine, modal_worker, args) -> bool:
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        return True

    if name == "/colors":
        print(format_colors_legend())
        return False

    if name == "/mood":
        if modal_worker:
            state = modal_worker.get_session_state.remote()
            traits = state.get("traits", {})
            tone = state.get("dominant_tone", "neutral")
        else:
            traits = engine.session.traits
            tone = engine.session.dominant_tone
        print(format_mood_summary(traits, tone))
        return False

    if name == "/refresh":
        _do_refresh(engine, modal_worker)
        return False

    if name == "/strength":
        try:
            val = float(arg) if arg else CHAT_HOOK_STRENGTH
        except ValueError:
            print("Usage: /strength 1.0")
            return False
        if modal_worker:
            modal_worker.set_strength.remote(val)
        else:
            engine.set_hook_strength(val)
        print(f"Hook strength set to {val}")
        return False

    if name == "/affect":
        scale_map = {"high": 2.0, "low": 0.0, "neutral": None}
        key = arg.lower() if arg else "neutral"
        scale = scale_map.get(key, None)
        if arg and key not in scale_map:
            print("Usage: /affect high|low|neutral")
            return False
        if modal_worker:
            modal_worker.set_manual_affect.remote(scale)
        else:
            engine.set_manual_affect(scale)
        print(f"Manual affect scale: {scale}")
        return False

    if name == "/save":
        path = _save_if_requested(engine, modal_worker, args, force=True)
        print(f"Saved session to {path}")
        return False

    print(f"Unknown command: {name}")
    return False


def _do_refresh(engine, modal_worker) -> None:
    if modal_worker:
        result = modal_worker.refresh_affect.remote(force=True)
    else:
        result = engine.refresh_affect(force=True)
    if not result.get("ok"):
        print(f"[refresh] skipped: {result.get('reason', 'unknown')}")
        return
    _print_shift_if_needed(result)
    print(format_affect_line(result.get("traits", {}), source=result.get("source", "")))


def _save_if_requested(engine, modal_worker, args, force: bool = False) -> str:
    out = Path(args.output)
    if modal_worker is not None:
        return modal_worker.save_session_artifact.remote(out.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = engine.session.to_log_dict()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotional CLI chat")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="Run on local CUDA GPU")
    mode.add_argument("--modal", action="store_true", help="Run via Modal worker")
    parser.add_argument("--strength", type=float, default=CHAT_HOOK_STRENGTH)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=str,
        default="data/artifacts/benchmarks/phase_chat.json",
    )
    args = parser.parse_args()

    if args.modal:
        return run_modal_cli(args)
    if args.local:
        return run_local_cli(args)

    # Default: try local CUDA, else modal
    import torch

    if torch.cuda.is_available():
        return run_local_cli(args)
    print("No local CUDA — falling back to Modal.")
    return run_modal_cli(args)


if __name__ == "__main__":
    sys.exit(main())
