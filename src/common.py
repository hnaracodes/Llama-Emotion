"""Modal app, container image, and shared volumes."""

from pathlib import Path

import modal

from src.config import (
    APP_NAME,
    MODEL_CACHE_DIR,
    VOLUME_NAME,
)

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# HF cache + project artifacts share one volume mount (Modal forbids duplicate mounts).
volume_mounts = {
    MODEL_CACHE_DIR: model_volume,
}

_hf_env = {
    "HF_HOME": MODEL_CACHE_DIR,
    "TRANSFORMERS_CACHE": MODEL_CACHE_DIR,
    "TOKENIZERS_PARALLELISM": "false",
    "SAA_RUNTIME": "modal",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BAKED_DATA = _PROJECT_ROOT / "data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.43.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "snntorch>=0.7.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "huggingface-hub>=0.24.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )
    .env(_hf_env)
)

_vllm_env = {
    **_hf_env,
    # Legacy engine required for kv_cache_dtype=fp8 on vLLM <0.11 (L4/Modal image).
    "VLLM_USE_V1": "0",
}

# Phase 1b vLLM serving (Python 3.12, separate heavy image)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.8.0,<0.11",
        "transformers>=4.45.0,<5.0.0",
        "huggingface-hub>=0.24.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )
    .env(_vllm_env)
)

# TRIBEv2 needs extra deps; use a separate image for affective pipeline
affective_image = image.pip_install(
    "pandas>=2.0.0",
    "einops>=0.7.0",
    "sentence-transformers>=3.0.0",
)
if (_BAKED_DATA / "scenarios").is_dir():
    affective_image = affective_image.add_local_dir(
        _BAKED_DATA / "scenarios",
        remote_path="/opt/saa/data/scenarios",
    )
if (_BAKED_DATA / "lexicon").is_dir():
    affective_image = affective_image.add_local_dir(
        _BAKED_DATA / "lexicon",
        remote_path="/opt/saa/data/lexicon",
    )

# Mount local package for Modal functions
image = image.add_local_python_source("src")
vllm_image = vllm_image.add_local_python_source("src")
affective_image = affective_image.add_local_python_source("src")


def hf_secret() -> modal.Secret:
    """Requires: modal secret create huggingface-secret HF_TOKEN=<your_hf_token>"""
    return modal.Secret.from_name("huggingface-secret")


def gpu_kwargs() -> dict:
    from src.config import GPU_TIMEOUT_SEC, GPU_TYPE

    return {
        "gpu": GPU_TYPE,
        "timeout": GPU_TIMEOUT_SEC,
        "volumes": volume_mounts,
        "secrets": [hf_secret()],
    }


def vllm_gpu_kwargs() -> dict:
    from src.config import GPU_TYPE, VLLM_GPU_TIMEOUT_SEC

    return {
        "gpu": GPU_TYPE,
        "timeout": VLLM_GPU_TIMEOUT_SEC,
        "volumes": volume_mounts,
        "secrets": [hf_secret()],
    }
