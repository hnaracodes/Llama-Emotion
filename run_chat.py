"""
Modal persistent chat worker — warm GPU with W4 Llama + affective hooks.

Usage:
  modal run run_chat.py --help
  .venv\\Scripts\\python.exe chat.py --modal
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from src.common import app, gpu_kwargs, image, model_volume
from src.config import ARTIFACTS_MOUNT, CHAT_HOOK_STRENGTH, MODEL_ID


@app.cls(image=image, **gpu_kwargs())
class EmotionalChatWorker:
    """Load model once; serve chat_turn and refresh_affect."""

    @modal.enter()
    def setup(self) -> None:
        from src.chat.engine import ChatEngine
        from src.llm.loader import load_quantized_llama

        self.model, self.tokenizer = load_quantized_llama()
        self.engine = ChatEngine(
            self.model,
            self.tokenizer,
            hook_strength=CHAT_HOOK_STRENGTH,
        )
        # Initial affect from empty/synthetic transcript seed
        self.engine.session.dominant_tone = "neutral"
        self.engine.session.traits = {
            "engagement": 0.0,
            "arousal": 0.0,
            "tension": 0.0,
            "warmth": 0.0,
            "stability": 1.0,
            "shift": 0.0,
        }

    @modal.method()
    def chat_turn(self, user_text: str, *, temperature: float = 0.7) -> dict[str, Any]:
        return self.engine.generate_reply(user_text, temperature=temperature)

    @modal.method()
    def refresh_affect(self, force: bool = True) -> dict[str, Any]:
        return self.engine.refresh_affect(force=force)

    @modal.method()
    def set_strength(self, strength: float) -> dict[str, Any]:
        self.engine.set_hook_strength(strength)
        return {"hook_strength": strength}

    @modal.method()
    def set_manual_affect(self, scale: float | None) -> dict[str, Any]:
        self.engine.set_manual_affect(scale)
        return {"manual_affect_scale": scale}

    @modal.method()
    def get_session_state(self) -> dict[str, Any]:
        return self.engine.session.to_log_dict()

    @modal.method()
    def save_session_artifact(self, filename: str = "phase_chat.json") -> str:
        out = Path(ARTIFACTS_MOUNT) / "benchmarks" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": MODEL_ID,
            **self.engine.session.to_log_dict(),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        model_volume.commit()
        return str(out)

    @modal.exit()
    def teardown(self) -> None:
        if hasattr(self, "engine"):
            self.engine.cleanup()


@app.local_entrypoint()
def main(user_text: str = "Hello, how are you?"):
    worker = EmotionalChatWorker()
    print(json.dumps(worker.chat_turn.remote(user_text), indent=2))
