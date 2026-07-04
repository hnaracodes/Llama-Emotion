#!/usr/bin/env python3
"""Download EmpatheticDialogues tar.gz and extract CSV splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EMPATHETICDIALOGUES_DIR, PROJECT_ROOT
from src.data.ensure import ensure_empatheticdialogues


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EmpatheticDialogues")
    parser.add_argument("--out-dir", type=Path, default=EMPATHETICDIALOGUES_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    ensure_empatheticdialogues(out, force=args.force)
    print(f"Done. CSV files in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
