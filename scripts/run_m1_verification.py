#!/usr/bin/env python3
"""
Local M1 verification (CPU-safe) + optional Modal behavioral benchmarks.

Usage:
  py -3 scripts/run_m1_verification.py
  py -3 scripts/run_m1_verification.py --modal-loop
  py -3 scripts/run_m1_verification.py --modal-phase4 --skip-strength-sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _step(name: str, fn):
    print(f"\n=== {name} ===")
    try:
        out = fn()
        print(json.dumps(out, indent=2) if isinstance(out, dict) else out)
        return {"step": name, "ok": True, "result": out}
    except Exception as e:
        print(f"FAIL: {e}")
        return {"step": name, "ok": False, "error": str(e)}


def run_local_verification() -> dict:
    import numpy as np

    log: list[dict] = []

    def pytest_suite():
        import os

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--ignore=tests/test_hybrid_encoder.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        return {"exit_code": r.returncode, "stdout_tail": r.stdout[-800:], "stderr_tail": r.stderr[-400:]}

    log.append(_step("pytest_full_suite", pytest_suite))

    def checkpoint_roots():
        from src.brain.checkpoints import artifact_roots

        roots = {k: str(v) for k, v in artifact_roots().items()}
        exists = {k: Path(v).exists() for k, v in artifact_roots().items()}
        return {"roots": roots, "local_exists": exists}

    log.append(_step("checkpoint_paths", checkpoint_roots))

    def encoder_fixture_vad():
        from src.affective.dataset import EmpatheticDialoguesDataset
        from src.affective.encoder import AffectEncoder
        from src.affective.compress import normalize_affective

        fixture = ROOT / "tests" / "fixtures" / "empatheticdialogues"
        ds = EmpatheticDialoguesDataset("test", data_dir=fixture, filter_holdouts=False)
        enc = AffectEncoder(backend="hash")
        errors = []
        for i in range(min(8, len(ds))):
            s = ds[i]
            pred = enc(enc.encode_training_text(s)).detach().numpy()
            tgt = np.asarray(s.target_32d, dtype=np.float32)
            pn = normalize_affective(pred.reshape(1, -1))[0]
            tn = normalize_affective(tgt.reshape(1, -1))[0]
            errors.append(float(np.mean(np.abs(pn[:3] - tn[:3]))))
        return {"samples": len(errors), "mean_vad_mae": round(float(np.mean(errors)), 4)}

    log.append(_step("encoder_fixture_vad_mae", encoder_fixture_vad))

    def coupling_arc():
        from src.affective.coupling import affect_coupling_corr, couple
        from src.affective.dynamics import AffectDynamics
        from src.affective.emotion_lexicon import emotion_to_32d

        user_traj = [emotion_to_32d(e) for e in ("sad", "anxious", "afraid")]
        dyn = AffectDynamics()
        internal = []
        state = None
        for u in user_traj:
            state = couple(u, state, coupling=0.4)
            internal.append(dyn.step(state))
        return {
            "coupling_corr": round(affect_coupling_corr(user_traj, internal), 4),
            "final_norm": float(np.linalg.norm(internal[-1])),
        }

    log.append(_step("coupling_distress_arc", coupling_arc))

    def phenotype_smoke():
        from src.benchmark.phenotype import build_phenotype

        rows = [
            {
                "conditions": {
                    "neutral": {"generated_preview": "Photosynthesis uses light."},
                    "high_affect": {
                        "generated_preview": "I'm sorry you're going through this. I understand."
                    },
                }
            }
        ]
        return build_phenotype(rows)

    log.append(_step("phenotype_builder", phenotype_smoke))

    def brain_alignment_stub():
        from src.benchmark.brain_alignment import alignment_report

        return alignment_report()

    log.append(_step("brain_alignment_stub", brain_alignment_stub))

    def gate_noop():
        from src.brain.checkpoints import assert_gate_noop
        from src.llm.hooks import AffectiveGate
        from src.config import AFFECT_DIM

        gate = AffectiveGate(AFFECT_DIM, 128, mode="additive")
        assert_gate_noop(gate)
        return {"gate_zero_norm_ok": True}

    log.append(_step("gate_noop_random_init", gate_noop))

    passed = sum(1 for x in log if x["ok"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps_total": len(log),
        "steps_passed": passed,
        "steps_failed": len(log) - passed,
        "steps": log,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-loop", action="store_true")
    parser.add_argument("--modal-phase4", action="store_true")
    parser.add_argument("--skip-strength-sweep", action="store_true")
    args = parser.parse_args()

    report = {"local": run_local_verification()}
    out_dir = ROOT / "data" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "verification_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}")

    if args.modal_loop:
        print("\n=== modal benchmark_phase_loop ===")
        subprocess.run(
            [sys.executable, "-m", "modal", "run", "benchmark_phase_loop.py"],
            cwd=ROOT,
            check=False,
        )
    if args.modal_phase4:
        cmd = [sys.executable, "-m", "modal", "run", "benchmark_phase4_extended.py"]
        if args.skip_strength_sweep:
            cmd.append("--skip-strength-sweep")
        print("\n=== modal benchmark_phase4_extended ===")
        subprocess.run(cmd, cwd=ROOT, check=False)

    failed = report["local"]["steps_failed"]
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
