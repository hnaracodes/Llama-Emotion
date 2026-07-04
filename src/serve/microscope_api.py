"""Emotion Microscope API — live affect introspection (Track E)."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Llama-Emotion Microscope")

_engine_factory: Callable[[], Any] | None = None
_sessions: dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


def set_engine_factory(factory: Callable[[], Any] | None) -> None:
    """Test hook: inject mock ChatEngine factory."""
    global _engine_factory
    _engine_factory = factory


def _get_engine(session_id: str):
    if session_id not in _sessions:
        if _engine_factory is not None:
            _sessions[session_id] = _engine_factory()
        else:
            from src.llm.loader import load_quantized_llama
            from src.chat.engine import ChatEngine

            model, tokenizer = load_quantized_llama()
            _sessions[session_id] = ChatEngine(model, tokenizer)
    return _sessions[session_id]


@app.get("/state/{session_id}")
def get_state(session_id: str = "default") -> dict[str, Any]:
    if session_id not in _sessions:
        return {"source": "none", "session_id": session_id, "ok": False}
    engine = _sessions[session_id]
    vec = engine.session.affect_vector
    return {
        "ok": True,
        "session_id": session_id,
        "source": engine._last_affect_source or "encoder:pending",
        "dominant_tone": engine.session.dominant_tone,
        "traits": dict(engine.session.traits),
        "turn_index": engine.session.turn_index,
        "affect_vector": vec.tolist() if vec is not None else None,
        "affect_trajectory": engine.session.affect_trajectory,
    }


@app.post("/reset/{session_id}")
def reset_session(session_id: str = "default") -> dict[str, str]:
    if session_id in _sessions:
        eng = _sessions.pop(session_id)
        eng.cleanup()
    return {"status": "reset", "session_id": session_id}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    engine = _get_engine(req.session_id)
    result = engine.generate_reply(req.message, return_introspection=True)
    vec = engine.session.affect_vector
    return {
        "reply": result["reply"],
        "source": result.get("affect_source", ""),
        "encoder_source": result.get("encoder_source"),
        "gate_source": result.get("gate_source"),
        "amygdala_source": result.get("amygdala_source"),
        "dominant_tone": result.get("dominant_tone"),
        "traits": result.get("traits"),
        "affect_vector": vec.tolist() if vec is not None else None,
        "introspection": result.get("introspection", {}),
    }
