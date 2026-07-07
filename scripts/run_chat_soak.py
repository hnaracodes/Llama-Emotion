#!/usr/bin/env python3
"""Run the 10-turn / 256-token chat soak benchmark on Modal (Windows-safe).

Usage:
  py -3 scripts/run_chat_soak.py
  py -3 scripts/run_chat_soak.py --no-fail-on-collapse
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Modal chat soak benchmark")
    p.add_argument(
        "--no-fail-on-collapse",
        action="store_true",
        help="Record results even if collapse is detected",
    )
    args = p.parse_args()

    cmd = [sys.executable, "-m", "modal", "run", "benchmark_phase_chat_soak.py"]
    if args.no_fail_on_collapse:
        cmd.append("--fail-on-collapse=False")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    out = _ROOT / "data" / "artifacts" / "phase_chat_soak_last_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
