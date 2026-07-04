"""Local entrypoint for Emotion Microscope API."""

from __future__ import annotations

import uvicorn

from src.serve.microscope_api import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
