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

# v1 supervision (§1 — post-TRIBEv2)
SUPERVISION_VERSION = "empatheticdialogues_v1"
EMPATHETICDIALOGUES_DIR = DATA_DIR / "raw" / "empatheticdialogues"
SCENARIO_HOLDOUT_DIR = DATA_DIR / "scenarios"
AFFECT_ENCODER_CKPT_NAME = "encoder.pt"
AFFECT_ENCODER_DIR = ARTIFACTS_DIR / "affect"
EMOTION_LEXICON_JSON = "emotion_lexicon.json"
AFFECT_ENCODER_TRAIN_LR = 1e-3
AFFECT_ENCODER_TRAIN_EPOCHS = 3
AFFECT_ENCODER_BATCH_SIZE = 64
AFFECT_ENCODER_BACKEND = "hybrid"  # hybrid (MiniLM+head) | hash (offline CI)
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_LEXICON_PATH = DATA_DIR / "lexicon" / "emotion_lexicon.json"
AFFECT_ENCODER_CONTRASTIVE_WEIGHT = 0.25
AMYGDALA_CKPT_NAME = "amygdala.pt"
GATE_CKPT_NAME = "affect_gate.pt"
SNN_CKPT_DIR = ARTIFACTS_DIR / "snn"
GATE_CKPT_DIR = ARTIFACTS_DIR / "gate"
GATE_TRAIN_MAX_SAMPLES = 500
GATE_TRAIN_EPOCHS = 3
GATE_GPU_TIMEOUT_SEC = 7200
GATE_NOOP_EPS = 1e-3
GATE_CONTRASTIVE_MARGIN = 0.5
GATE_REPETITION_WEIGHT = 0.25
GATE_MAX_AFFECT_NORM = 1.0
GATE_HOLDOUT_EVAL_EVERY = 50
GATE_COLLAPSE_MAX_RUN = 8
DEFAULT_REPETITION_PENALTY = 1.12
# Gate v3 — listener CE + balanced batches
# v3.1 bumps the tag after the checkpoint-selection/loss-leak/frozen-Llama
# hardening pass so checkpoints trained before/after that fix are distinguishable.
GATE_VERSION = "v3.1_listener_ce_hardened"
GATE_V3_LISTENER_MAX_TOKENS = 128
GATE_NEUTRAL_BATCH_RATIO = 0.5
GATE_DISTRESS_MARGIN = 0.1
GATE_NEUTRAL_CE_EPS = 0.05
GATE_HOLDOUT_EVERY = 50
GATE_HOLDOUT_MAX_NEW_TOKENS = 96
GATE_EMPATHY_ID_WEIGHT = 0.0
GATE_DISTRESS_EMOTIONS = frozenset(
    {"anxious", "sad", "afraid", "terrified", "devastated", "distress"}
)
GATE_NEUTRAL_EMOTIONS = frozenset(
    {"neutral", "content", "prepared", "confident", "grateful", "faithful", "impressed"}
)
# Deliberately disjoint wording from PHASE4_ABLATION_PROMPTS: these are used only
# to pick the best gate checkpoint *during* training. If checkpoint selection used
# the same prompts as the Phase 4 ablation report, that report would no longer be
# an independent check of the saved gate.
GATE_TRAIN_HOLDOUT_PROMPTS = [
    {
        "id": "gate_holdout_distress",
        "prompt": (
            "I've been feeling really anxious about work lately and I don't "
            "know how to cope. Can you help?"
        ),
    },
    {
        "id": "gate_holdout_neutral",
        "prompt": "In one sentence, explain what causes ocean tides.",
    },
]
AFFECT_MEMBRANE_RESET_TURNS = 32
AFFECT_DECAY = 0.85
AFFECT_GAIN = 0.35
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

# Chat hardening (docs/chat_hardening_plan.md)
# Safe, generic reply returned when a chat turn collapses even after the
# hooks-off retry — never emits raw model output once collapse is confirmed.
CHAT_COLLAPSE_FALLBACK_REPLY = (
    "I want to make sure I respond thoughtfully here — could you say that "
    "one more time, maybe with a bit more detail?"
)
CHAT_LOG_SCHEMA_VERSION = 2

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
