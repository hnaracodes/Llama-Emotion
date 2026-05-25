"""Resolve project .venv executables — never use global Python/Modal."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def venv_python(root: Path | None = None) -> Path:
    """Return path to project .venv Python interpreter."""
    root = root or PROJECT_ROOT
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",  # Windows
        root / ".venv" / "bin" / "python",  # macOS / Linux
    ]
    for exe in candidates:
        if exe.is_file():
            return exe
    raise FileNotFoundError(
        f"No .venv found under {root}. "
        "Create one in the project root: python -m venv .venv"
    )


def venv_modal_cmd(*args: str, root: Path | None = None) -> list[str]:
    """Build command: .venv python -m modal [args]."""
    py = venv_python(root)
    return [str(py), "-m", "modal", *args]


def venv_pytest_cmd(*extra: str, root: Path | None = None) -> list[str]:
    """Build command: .venv python -m pytest tests/ [extra]."""
    py = venv_python(root)
    return [str(py), "-m", "pytest", "tests/", *extra]


def ensure_venv_or_exit() -> Path:
    """Exit with clear message if .venv is missing."""
    try:
        return venv_python()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
