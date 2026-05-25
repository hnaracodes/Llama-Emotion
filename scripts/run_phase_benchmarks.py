#!/usr/bin/env python3
"""
Orchestrate Spiking Affective Adapter phase benchmarks and emit human-readable reports.

Usage (always via project .venv):
  .venv\\Scripts\\python.exe scripts/run_phase_benchmarks.py --local-only
  .venv\\Scripts\\python.exe scripts/run_phase_benchmarks.py --phases 1a,1b --skip-8192
  .venv\\Scripts\\python.exe scripts/run_phase_benchmarks.py --phases 1a,1b,1b-vllm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from venv_tools import ensure_venv_or_exit, venv_modal_cmd, venv_pytest_cmd  # noqa: E402

VENV_PYTHON = ensure_venv_or_exit()


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_pytest() -> dict[str, Any]:
    code, out, err = _run(venv_pytest_cmd("-q"), timeout=300)
    combined = (out + err).strip()
    passed = failed = 0
    for line in combined.splitlines():
        if " passed" in line and " in " in line:
            parts = line.replace(" failed", "").replace(" passed", "").split(",")
            for p in parts:
                p = p.strip()
                if p.endswith("passed"):
                    try:
                        passed = int(p.split()[0])
                    except ValueError:
                        pass
                if "failed" in line and "failed" in p:
                    try:
                        failed = int(p.split()[0])
                    except ValueError:
                        pass
    return {
        "exit_code": code,
        "passed": passed,
        "failed": failed,
        "output_tail": combined[-2000:] if combined else "",
        "ok": code == 0,
    }


def run_modal_phase(script: str, extra_args: list[str] | None = None, *, timeout: int = 1800) -> dict[str, Any]:
    cmd = venv_modal_cmd("run", script, *(extra_args or []))
    code, out, err = _run(cmd, timeout=timeout)
    payload: dict[str, Any] = {
        "script": script,
        "exit_code": code,
        "ok": code == 0,
        "stderr_tail": err[-3000:] if err else "",
    }
    # Modal prints JSON result at end of stdout
    for block in _extract_json_blocks(out):
        if isinstance(block, dict) and ("runs" in block or "analytic_kv_table" in block or "hf_quantized_cache_benchmarks" in block):
            payload["result"] = block
            break
    if "result" not in payload:
        payload["stdout_tail"] = out[-4000:] if out else ""
    return payload


def _extract_json_blocks(text: str) -> list[Any]:
    blocks: list[Any] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : i + 1]
                try:
                    blocks.append(json.loads(snippet))
                except json.JSONDecodeError:
                    pass
                start = None
    return blocks


def evaluate_phase1a(result: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    runs = result.get("runs", {})
    w4 = runs.get("w4_nf4", {})
    if w4.get("peak_vram_gb", 0) <= 0:
        notes.append("FAIL: W4 peak_vram_gb missing or zero")
    else:
        notes.append(f"PASS: W4 peak VRAM = {w4.get('peak_vram_gb')} GB")
    ratio = result.get("vram_reduction_ratio")
    if ratio is not None:
        if ratio >= 1.5:
            notes.append(f"PASS: VRAM reduction ratio = {ratio}x (FP16 / W4)")
        else:
            notes.append(f"WARN: VRAM reduction ratio only {ratio}x (expected ≥1.5)")
    preview = (w4.get("generated_preview") or "")[:80]
    if preview.strip():
        notes.append(f"PASS: Generation preview present ({preview!r}…)")
    else:
        notes.append("WARN: Empty generation preview")
    return notes


def evaluate_phase1b(result: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for run in result.get("hf_quantized_cache_benchmarks", []):
        target = run.get("target_seq_len")
        actual = run.get("actual_seq_len")
        if actual == target:
            notes.append(f"PASS: seq {target} — actual_seq_len matches target")
        else:
            notes.append(f"WARN: seq {target} — actual_seq_len={actual} (expected {target})")

        modes = run.get("modes", {})
        fp16 = modes.get("fp16_dynamic", {})
        int8 = modes.get("int8_quantized_storage", {})
        fp16_mb = fp16.get("kv_storage_mb", 0)
        int8_mb = int8.get("kv_storage_mb", 0)
        if fp16_mb > 0:
            notes.append(f"PASS: seq {target} FP16 kv_storage_mb = {fp16_mb}")
        else:
            notes.append(f"FAIL: seq {target} FP16 kv_storage_mb = 0 (reporting bug)")
        if int8_mb > 0 and fp16_mb > 0 and int8_mb < fp16_mb:
            notes.append(f"PASS: seq {target} INT8 storage ({int8_mb} MB) < FP16 ({fp16_mb} MB)")
        elif fp16_mb > 0:
            notes.append(f"WARN: seq {target} INT8 ({int8_mb} MB) not smaller than FP16 ({fp16_mb} MB)")

        previews = {m.get("decode_tail_preview") for m in modes.values()}
        if len(previews) == 1:
            notes.append(f"PASS: seq {target} — all modes same decode tail")
        else:
            notes.append(f"WARN: seq {target} — decode tails differ across modes: {previews}")
    return notes


def format_analytic_table(rows: list[dict]) -> str:
    lines = ["| seq_len | FP16 GB | INT8 GB | INT4 GB |", "|---------|---------|---------|---------|"]
    for r in rows:
        lines.append(
            f"| {r['seq_len']} | {r['kv_fp16_gb']} | {r['kv_int8_gb']} | {r['kv_int4_gb']} |"
        )
    return "\n".join(lines)


def format_hf_table(runs: list[dict]) -> str:
    lines = [
        "| target | actual | mode | kv_storage_mb | peak_vram_gb | time_s |",
        "|--------|--------|------|---------------|--------------|--------|",
    ]
    for run in runs:
        target = run.get("target_seq_len")
        actual = run.get("actual_seq_len")
        for name, mode in run.get("modes", {}).items():
            lines.append(
                f"| {target} | {actual} | {name} | {mode.get('kv_storage_mb')} | "
                f"{mode.get('peak_vram_gb')} | {mode.get('prefill_plus_decode_sec')} |"
            )
    return "\n".join(lines)


def build_report(
    *,
    pytest_result: dict[str, Any],
    phase_results: dict[str, dict[str, Any]],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"# Phase Research Report — {ts}", ""]

    # Executive summary
    modal_ok = all(p.get("ok") for p in phase_results.values() if p)
    local_ok = pytest_result.get("ok", False)
    overall = "PASS" if local_ok and modal_ok else "PARTIAL" if modal_ok or local_ok else "FAIL"
    sections.append("## Executive summary")
    sections.append(
        f"Overall: **{overall}**. Local pytest: {pytest_result.get('passed', '?')} passed. "
        f"Modal phases run: {', '.join(phase_results.keys()) or 'none'}."
    )
    sections.append("")

    sections.append("## Local tests")
    status = "PASS" if pytest_result.get("ok") else "FAIL"
    sections.append(f"- pytest: **{status}** ({pytest_result.get('passed', 0)} passed, {pytest_result.get('failed', 0)} failed)")
    if not pytest_result.get("ok"):
        sections.append(f"\n```\n{pytest_result.get('output_tail', '')}\n```")
    sections.append("")

    if "1a" in phase_results:
        p = phase_results["1a"]
        sections.append("## Phase 1a — W4 NF4 weights")
        if not p.get("ok"):
            sections.append(f"**FAILED** (exit {p.get('exit_code')})")
            sections.append(f"```\n{p.get('stderr_tail') or p.get('stdout_tail', '')}\n```")
        else:
            res = p.get("result", {})
            w4 = res.get("runs", {}).get("w4_nf4", {})
            fp = res.get("runs", {}).get("fp16_baseline", {})
            sections.append("| Run | Peak VRAM (GB) | Tokens/sec |")
            sections.append("|-----|----------------|------------|")
            sections.append(f"| W4 NF4 | {w4.get('peak_vram_gb', 'n/a')} | {w4.get('tokens_per_sec', 'n/a')} |")
            if fp:
                sections.append(f"| FP16 baseline | {fp.get('peak_vram_gb', 'n/a')} | {fp.get('tokens_per_sec', 'n/a')} |")
            if res.get("vram_reduction_ratio"):
                sections.append(f"\n**VRAM reduction ratio:** {res['vram_reduction_ratio']}x")
            sections.append("\n**Evaluation:**")
            for note in evaluate_phase1a(res):
                sections.append(f"- {note}")
        sections.append("")

    if "1b" in phase_results:
        p = phase_results["1b"]
        sections.append("## Phase 1b — KV cache (HF)")
        if not p.get("ok"):
            sections.append(f"**FAILED** (exit {p.get('exit_code')})")
            sections.append(f"```\n{p.get('stderr_tail') or p.get('stdout_tail', '')}\n```")
        else:
            res = p.get("result", {})
            sections.append("### Analytic (theory)")
            sections.append(format_analytic_table(res.get("analytic_kv_table", [])))
            sections.append("\n### Live HF benchmarks")
            sections.append(format_hf_table(res.get("hf_quantized_cache_benchmarks", [])))
            sections.append("\n**Evaluation:**")
            for note in evaluate_phase1b(res):
                sections.append(f"- {note}")
        sections.append("")

    if "1b-vllm" in phase_results:
        p = phase_results["1b-vllm"]
        sections.append("## Phase 1b — KV cache (vLLM)")
        if not p.get("ok"):
            sections.append(f"**FAILED** (exit {p.get('exit_code')})")
        else:
            res = p.get("result", {})
            vllm = res.get("vllm_kv_benchmark", res)
            for name, mode in vllm.get("modes", {}).items():
                sections.append(f"- **{name}**: status={mode.get('status')}, peak={mode.get('peak_vram_gb')} GB, latency={mode.get('latency_sec')}s")
        sections.append("")

    sections.append("## Recommendations")
    if not pytest_result.get("ok"):
        sections.append("- Fix failing local unit tests before GPU runs.")
    if "1b" not in phase_results:
        sections.append("- Run Phase 1b HF: `.venv\\Scripts\\python.exe -m modal run benchmark_phase1b.py`")
    if "1b-vllm" not in phase_results:
        sections.append("- Run Phase 1b vLLM: `.venv\\Scripts\\python.exe -m modal run benchmark_phase1b_vllm.py`")
    sections.append("- Next: Phase 4 hybrid — `.venv\\Scripts\\python.exe -m modal run run_hybrid.py`")

    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAA phase benchmarks and print report")
    parser.add_argument(
        "--phases",
        default="1a,1b",
        help="Comma-separated: 1a, 1b, 1b-vllm, 2, 4",
    )
    parser.add_argument("--local-only", action="store_true", help="Skip Modal GPU runs")
    parser.add_argument("--skip-8192", action="store_true", help="Pass --skip-8192 to Phase 1b")
    parser.add_argument("--output", type=Path, help="Write report markdown to file")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    pytest_result = run_pytest()
    phase_results: dict[str, dict[str, Any]] = {}

    if not args.local_only:
        if "1a" in phases:
            phase_results["1a"] = run_modal_phase("benchmark_phase1a.py", timeout=1200)
        if "1b" in phases:
            extra = ["--skip-8192"] if args.skip_8192 else []
            phase_results["1b"] = run_modal_phase("benchmark_phase1b.py", extra, timeout=2400)
        if "1b-vllm" in phases:
            phase_results["1b-vllm"] = run_modal_phase("benchmark_phase1b_vllm.py", timeout=1800)

    report = build_report(pytest_result=pytest_result, phase_results=phase_results)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    all_ok = pytest_result.get("ok") and all(p.get("ok") for p in phase_results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
