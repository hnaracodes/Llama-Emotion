"""TRIBEv2 client with synthetic fallback for offline / CI testing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.affective.compress import AffectiveCompressor, normalize_affective
from src.config import AFFECT_DIM, TRIBE_ID
from src.encoder.delta_mod import delta_modulate, spike_sparsity


def synthetic_fmri_timeseries(
    n_timesteps: int = 100,
    n_vertices: int = 1024,
    seed: int = 42,
) -> np.ndarray:
    """Proxy fMRI-like signal when TRIBEv2 is unavailable."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    base = np.sin(t)[:, None] * rng.standard_normal((1, n_vertices))
    noise = 0.1 * rng.standard_normal((n_timesteps, n_vertices))
    return (base + noise).astype(np.float32)


def run_tribev2_from_transcript(
    messages: list,
    *,
    cache_folder: str = "./cache",
    temp_dir: str | None = None,
) -> Tuple[np.ndarray, str]:
    """
    Run TRIBEv2 on a chat transcript (list of ChatMessage or role/content dicts).

    Writes a temporary timeline text file and calls run_tribev2_predict(text_path=...).
    """
    import tempfile
    from pathlib import Path

    from src.chat.transcript import format_tribev2_transcript

    if messages and hasattr(messages[0], "role"):
        t0 = messages[0].timestamp if hasattr(messages[0], "timestamp") else None
        text = format_tribev2_transcript(messages, session_start=t0)  # type: ignore[arg-type]
    else:
        # dict fallback
        from src.chat.session import ChatMessage
        import time

        now = time.time()
        cms = [
            ChatMessage(role=m["role"], content=m["content"], timestamp=now + i)
            for i, m in enumerate(messages)
        ]
        text = format_tribev2_transcript(cms)

    if temp_dir:
        path = Path(temp_dir) / "chat_transcript.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return run_tribev2_predict(text_path=str(path), cache_folder=cache_folder)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        temp_path = f.name
    try:
        return run_tribev2_predict(text_path=temp_path, cache_folder=cache_folder)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def run_tribev2_predict(
    *,
    video_path: Optional[str] = None,
    text_path: Optional[str] = None,
    cache_folder: str = "./cache",
) -> Tuple[np.ndarray, str]:
    """
    Run TRIBEv2 if installed; else synthetic fallback.

    Returns:
        preds: (T, V) float32
        source: description of data source
    """
    try:
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained(TRIBE_ID, cache_folder=cache_folder)
        kwargs = {}
        if video_path:
            kwargs["video_path"] = video_path
        if text_path:
            kwargs["text_path"] = text_path
        if not kwargs:
            raise ValueError("Provide video_path or text_path for TRIBEv2")

        df = model.get_events_dataframe(**kwargs)
        preds, _ = model.predict(events=df)
        arr = np.asarray(preds, dtype=np.float32)
        return arr, f"tribev2:{TRIBE_ID}"
    except Exception as exc:
        arr = synthetic_fmri_timeseries()
        return arr, f"synthetic_fallback:{exc.__class__.__name__}"


def pipeline_to_spikes(
    fmri_ts: np.ndarray,
    *,
    theta: float,
    affect_dim: int = AFFECT_DIM,
    compressor: Optional[AffectiveCompressor] = None,
) -> dict:
    """fMRI (T,V) → 32-d affective vectors → delta-mod spikes."""
    if compressor is None:
        compressor = AffectiveCompressor(n_components=affect_dim)
        compressor.fit_roi_groups(fmri_ts.shape[1])

    affective = normalize_affective(compressor.transform(fmri_ts))
    spikes = delta_modulate(affective, theta=theta)
    return {
        "affective_vectors": affective,
        "spikes": spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes),
        "spike_sparsity": spike_sparsity(spikes),
        "T": affective.shape[0],
        "D": affective.shape[1],
    }


def save_pipeline_artifacts(
    out_dir: Path,
    fmri_ts: np.ndarray,
    pipeline_result: dict,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fmri_ts.npy", fmri_ts)
    np.save(out_dir / "affective_vectors.npy", pipeline_result["affective_vectors"])
    np.save(out_dir / "spikes.npy", pipeline_result["spikes"])
    import json

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
