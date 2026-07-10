"""
Modal web deploy — warm GPU ChatEngine + FastAPI + optional React static UI.

Deploy:
  cd web && npm install && npm run build
  py -3 -m modal deploy run_web_serve.py

Local API only (no Modal):
  py -3 run_microscope.py
  cd web && npm run dev
"""

from __future__ import annotations

from pathlib import Path

import modal

from src.common import app, gpu_kwargs, image
from src.config import CHAT_HOOK_STRENGTH

_PROJECT_ROOT = Path(__file__).resolve().parent
_WEB_DIST = _PROJECT_ROOT / "web" / "dist"
_WEB_REMOTE = "/opt/saa/web/dist"

# SPA files must be the last image layer (no build steps after add_local_*).
_serve_image = image
if (_WEB_DIST / "index.html").is_file():
    _serve_image = image.add_local_dir(_WEB_DIST, remote_path=_WEB_REMOTE)


@app.cls(
    image=_serve_image,
    **gpu_kwargs(),
    scaledown_window=600,
)
@modal.concurrent(max_inputs=8)
class WebChatServer:
    """Single warm GPU container serving HTTP chat + affect introspection."""

    @modal.enter()
    def setup(self) -> None:
        from src.chat.engine import ChatEngine
        from src.llm.loader import load_quantized_llama
        from src.serve.microscope_api import set_shared_engine

        self.model, self.tokenizer = load_quantized_llama()
        self.engine = ChatEngine(
            self.model,
            self.tokenizer,
            hook_strength=CHAT_HOOK_STRENGTH,
        )
        health = self.engine.gate_health()
        if health["source"] != "trained":
            raise RuntimeError(
                "WebChatServer refusing to serve: " f"{health['warning']}"
            )
        if health["warning"]:
            print(f"[WebChatServer] WARNING: {health['warning']}")
        set_shared_engine(self.engine)

    @modal.asgi_app()
    def web(self):
        from fastapi.staticfiles import StaticFiles

        from src.serve.microscope_api import app as api

        static_root = Path(_WEB_REMOTE)
        if static_root.is_dir() and (static_root / "index.html").is_file():
            api.mount(
                "/",
                StaticFiles(directory=str(static_root), html=True),
                name="spa",
            )
        return api

    @modal.exit()
    def teardown(self) -> None:
        from src.serve.microscope_api import set_shared_engine

        set_shared_engine(None)
        if hasattr(self, "engine"):
            self.engine.cleanup()


@app.local_entrypoint()
def main():
    print(
        "Deploy with: py -3 -m modal deploy run_web_serve.py\n"
        "Then open the URL printed by Modal (serves API + built web/dist if present)."
    )
