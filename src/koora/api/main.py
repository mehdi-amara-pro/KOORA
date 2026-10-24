"""
KOORA API. Day-1 version: health check, contract info, a demo insight, and the static web app.

Run locally:
    uvicorn koora.api.main:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from koora import __version__
from koora.contracts.schemas import SCHEMA_VERSION, Insight, _examples

app = FastAPI(title="KOORA API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "schema_version": SCHEMA_VERSION}


@app.get("/api/demo/insight", response_model=Insight)
def demo_insight() -> Insight:
    """A contract-valid example insight, so the frontend has something real-shaped to render."""
    return _examples()["insight"]


# Static web app, served from the same Container App for now.
# Mounted last so /health and /api/* take precedence.
WEB_DIR = Path(os.environ.get("WEB_DIR", Path(__file__).resolve().parents[2] / "web"))
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
