"""Ensure training corpora exist (local disk or Modal volume)."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

TAR_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

_MEMBERS = {
    "empatheticdialogues/train.csv": "train.csv",
    "empatheticdialogues/valid.csv": "valid.csv",
    "empatheticdialogues/test.csv": "test.csv",
}

_ALIASES = {
    "train.csv": "empchat_train.csv",
    "valid.csv": "empchat_valid.csv",
    "test.csv": "empchat_test.csv",
}


def _split_ready(out_dir: Path) -> bool:
    for names in (("train.csv",), ("empchat_train.csv",)):
        if (out_dir / names[0]).is_file():
            return True
    return False


def ensure_empatheticdialogues(out_dir: Path, *, force: bool = False) -> Path:
    """
    Download EmpatheticDialogues tar.gz if CSV splits are missing.

    Idempotent: skips download when train split already exists unless force=True.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force and _split_ready(out_dir):
        return out_dir

    tar_path = out_dir / "empatheticdialogues.tar.gz"
    urllib.request.urlretrieve(TAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member_name, local_name in _MEMBERS.items():
            extracted = False
            for member in tar.getmembers():
                if member.name.replace("\\", "/").endswith(member_name):
                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        continue
                    dest = out_dir / local_name
                    dest.write_bytes(extracted_file.read())
                    alias = _ALIASES.get(local_name)
                    if alias:
                        (out_dir / alias).write_bytes(dest.read_bytes())
                    extracted = True
                    break
            if not extracted:
                raise FileNotFoundError(f"Member {member_name} not found in archive")
    tar_path.unlink(missing_ok=True)
    return out_dir
