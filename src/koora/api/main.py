"""
KOORA API: health check, contract info, demo insight, simulated match replays, and the web app.

Run locally:
    uvicorn koora.api.main:app --reload
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from koora import __version__
from koora.api.replay import build_replay
from koora.contracts.schemas import SCHEMA_VERSION, Insight, _examples

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("KOORA_DATA_DIR", REPO_ROOT / "data" / "matches"))
MATCH_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

app = FastAPI(title="KOORA API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "schema_version": SCHEMA_VERSION}


@app.get("/api/demo/insight", response_model=Insight)
def demo_insight() -> Insight:
    """A contract-valid example insight, so the frontend has something real-shaped to render."""
    return _examples()["insight"]


@app.get("/api/matches")
def list_matches() -> list[dict]:
    """Simulated matches saved in data/matches, newest first."""
    if not DATA_DIR.is_dir():
        return []
    files = sorted(DATA_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"id": f.stem, "size_mb": round(f.stat().st_size / 1e6, 1)} for f in files]


@app.get("/api/matches/{match_id}/replay")
def match_replay(match_id: str) -> dict:
    """Compact replay of one match for the 3D viewer."""
    if not MATCH_ID.match(match_id):
        raise HTTPException(status_code=400, detail="Invalid match id")
    path = DATA_DIR / f"{match_id}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Match not found")
    return build_replay(path)


# Static web app, served from the same process for now.
# Mounted last so /health and /api/* take precedence.
WEB_DIR = Path(os.environ.get("WEB_DIR", REPO_ROOT / "src" / "web"))
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
