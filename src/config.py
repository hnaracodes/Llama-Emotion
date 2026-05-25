"""Central hyperparameters and paths for the Spiking Affective Adapter."""

from pathlib import Path

# Repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# LLM
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID_3B = "meta-llama/Llama-3.2-3B-Instruct"

# Affective / neuromorphic
TRIBE_ID = "facebook/tribev2"
AFFECT_DIM = 32
DELTA_THETA = 0.1
SNN_HIDDEN = 64
SNN_BETA = 0.9
SNN_THRESHOLD = 1.0

# Modal
APP_NAME = "spiking-affective-adapter"
VOLUME_NAME = "saa-models"
MODEL_CACHE_DIR = "/models"
# Artifacts live on the same Modal volume root as the HF cache (single mount).
ARTIFACTS_MOUNT = "/models"
GPU_TYPE = "L4"
GPU_TIMEOUT_SEC = 600
VLLM_GPU_TIMEOUT_SEC = 900

# vLLM Phase 1b serving benchmarks
VLLM_KV_CACHE_DTYPES = ["auto", "fp8"]
VLLM_MAX_MODEL_LEN = 4096
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_MAX_NEW_TOKENS = 32

# Benchmark defaults
BENCHMARK_PROMPT = (
    "The amygdala processes emotional salience. In one sentence, explain "
    "why neuromorphic modulation might change language model outputs."
)
BENCHMARK_MAX_NEW_TOKENS = 64
BENCHMARK_CONTEXT_LENGTHS = [512, 2048, 8192]

# Generation
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
