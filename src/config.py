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

# Emotional CLI chat
AFFECT_REFRESH_SEC = 300
AFFECT_EMA_ALPHA = 0.35
CHAT_MAX_HISTORY_TOKENS = 2048
CHAT_HOOK_STRENGTH = 1.0
CHAT_KV_BITS = 8
CHAT_MAX_NEW_TOKENS = 256
CHAT_ASSISTANT_LABEL = "Amygdala"
TONE_SHIFT_THRESHOLD = 0.15
TONE_USE_COLOR = True
TONE_FLASH_ON_SHIFT = True

# Phase 4 extended — multi-prompt ablation + strength sweep
PHASE4_STRENGTH_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0]
PHASE4_ABLATION_PROMPTS = [
    {
        "id": "amygdala_explain",
        "prompt": BENCHMARK_PROMPT,
    },
    {
        "id": "emotional_support",
        "prompt": (
            "I'm feeling overwhelmed and alone. In two sentences, "
            "what would you say to me?"
        ),
    },
    {
        "id": "factual_brief",
        "prompt": "In one sentence, what is photosynthesis?",
    },
    {
        "id": "creative_tone",
        "prompt": (
            "Write one short sentence describing a rainy evening in a city."
        ),
    },
    {
        "id": "conflict_deescalation",
        "prompt": (
            "A friend is angry at me for missing their birthday. "
            "How should I respond in two sentences?"
        ),
    },
]

# Chat A/B — transcript-conditioned affect scenarios
CHAT_AB_USER_QUESTION = "What should I do next?"
CHAT_AB_TRANSCRIPTS: dict[str, list[dict[str, str]]] = {
    "distress": [
        {"role": "user", "content": "I failed my exam today and I feel awful."},
        {
            "role": "assistant",
            "content": "That sounds really hard. I'm here with you.",
        },
        {"role": "user", "content": "I don't think I can recover from this."},
    ],
    "neutral": [
        {"role": "user", "content": "Hi, I'm planning my week."},
        {"role": "assistant", "content": "Sure — what are your priorities?"},
        {"role": "user", "content": "Mostly work tasks and a gym session."},
    ],
    "hopeful": [
        {"role": "user", "content": "I failed my exam today and I feel awful."},
        {
            "role": "assistant",
            "content": "That sounds really hard. I'm here with you.",
        },
        {
            "role": "user",
            "content": "Actually I'm starting to feel a bit hopeful about retaking it.",
        },
    ],
}
