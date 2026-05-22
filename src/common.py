"""Modal app, container image, and shared volumes."""

import modal

from src.config import (
    APP_NAME,
    ARTIFACTS_MOUNT,
    MODEL_CACHE_DIR,
    VOLUME_NAME,
)

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# HF cache + project artifacts on the volume
volume_mounts = {
    MODEL_CACHE_DIR: model_volume,
    ARTIFACTS_MOUNT: model_volume,
}

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
        "lmdeploy>=0.6.0",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": MODEL_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# TRIBEv2 needs extra deps; use a separate image for affective pipeline
affective_image = image.pip_install(
    "pandas>=2.0.0",
    "einops>=0.7.0",
)

# Mount local package for Modal functions
image = image.add_local_python_source("src")
affective_image = affective_image.add_local_python_source("src")


def hf_secret() -> modal.Secret:
    """Requires: modal secret create saa-hf-secret HF_TOKEN=<your_hf_token>"""
    return modal.Secret.from_name("saa-hf-secret")


def gpu_kwargs() -> dict:
    from src.config import GPU_TIMEOUT_SEC, GPU_TYPE

    return {
        "gpu": GPU_TYPE,
        "timeout": GPU_TIMEOUT_SEC,
        "volumes": volume_mounts,
        "secrets": [hf_secret()],
    }
