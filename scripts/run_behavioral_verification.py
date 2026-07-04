#!/usr/bin/env python3
"""Run M1 behavioral verification on Modal after gate training.

Usage:
  py -3 scripts/run_behavioral_verification.py --skip-train
  py -3 scripts/run_behavioral_verification.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _modal(*args: str) -> dict | list:
    import os

    cmd = [sys.executable, "-m", "modal", "run", *args]
    print("+", " ".join(cmd), flush=True)
    # Modal's CLI writes rich-formatted output (checkmarks, box-drawing, emoji)
    # that isn't representable in the Windows default locale codec (cp1252).
    # `text=True` without an explicit encoding uses that locale codec for the
    # subprocess reader threads and crashes with UnicodeDecodeError before any
    # results are ever collected (found via a real --skip-train run on
    # Windows that always failed silently at this exact point). Force UTF-8
    # with lossy replacement instead of erroring.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    # Modal prints JSON result to stdout (last block)
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"stdout": text}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    results: dict = {}

    if not args.skip_train:
        print("=== Gate training ===", flush=True)
        results["gate_train"] = _modal(
            "train_gate.py",
            f"--max-samples={args.max_samples}",
            f"--epochs={args.epochs}",
        )

    print("=== Phase 4 extended ===", flush=True)
    results["phase4"] = _modal("benchmark_phase4_extended.py", "--skip-strength-sweep")

    print("=== Scenario holdout (2 highlights) ===", flush=True)
    results["scenarios"] = _modal(
        "benchmark_phase_scenarios.py",
        "--scenario-ids",
        "tone_calm_to_panic,factual_week_planning",
    )

    print("=== Chat A/B ===", flush=True)
    results["chat_ab"] = _modal("benchmark_phase_chat_ab.py")

    out = _ROOT / "data" / "artifacts" / "behavioral_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    from src.config import GATE_VERSION  # noqa: E402 (needs sys.path fixup above)

    summary = {
        "gate_version": GATE_VERSION,
        "phase4": results.get("phase4", {}),
        "scenarios": results.get("scenarios", {}),
        "chat_ab": results.get("chat_ab", {}),
    }
    if isinstance(summary["phase4"], dict):
        summary["phase4_fraction_text_changed"] = (
            summary["phase4"].get("summary", {}).get("fraction_text_changed")
        )
    summary_path = _ROOT / "data" / "artifacts" / "behavioral_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
