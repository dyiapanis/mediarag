"""MediaRAG dashboard backend — search + status routes for the web dashboard.

Runs inside the dashboard's plugin host, a separate process from the gateway:
the engine layers load once, on the first search request. Corpus root resolves
from the plugin dir's mediarag-config.yaml — the same chain the native tools
use, so dashboard and agents always see the same index.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys

from fastapi import APIRouter, HTTPException

router = APIRouter()

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ("text", "image", "audio")


# ponytail: flat `root:` line parse (config > env > default) in-place — the
# dashboard host has no plugin-context env, so the chain is walked locally.
def _root() -> str:
    cfg = os.path.join(PLUGIN_DIR, "mediarag-config.yaml")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                if line.startswith("root:"):
                    return line.split(":", 1)[1].strip().strip('"')
    return os.environ.get("MEDIARAG_ROOT") or os.path.expanduser("~/.mediarag")


@functools.lru_cache(maxsize=1)
def _tools():
    """Import the deployed plugin package once, from its own directory."""
    if PLUGIN_DIR not in sys.path:
        sys.path.insert(0, PLUGIN_DIR)
    from core import tools  # noqa: PLC0415 (host-side lazy import: engine loads on demand)

    return tools


@router.get("/status")
async def status():
    t = _tools()
    return await asyncio.to_thread(lambda: t.tool_status(root=_root()))


@router.get("/search")
async def search(q: str, kind: str | None = None, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    if kind and kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {KINDS}")
    k = min(max(k, 1), 20)
    t = _tools()
    return await asyncio.to_thread(
        lambda: t.tool_search(q.strip(), k=k, kind=kind or None, root=_root()))
