"""MediaRAG dashboard backend — search + status routes for the web dashboard.

The dashboard's plugin host runs under the dashboard process's own Python,
which does not carry the mediarag engine deps (transformers/onnxruntime live
in the installs env venvs). So this backend never imports the engine itself:
it spawns the deployed stdio server (scripts/mcp_server.py) under a resolved
engine interpreter and talks newline-JSON-RPC to it — the same surface agents
use, long-lived so the model loads once.
"""
from __future__ import annotations

import asyncio
import functools
import glob
import json
import os
import subprocess

from fastapi import APIRouter, HTTPException

router = APIRouter()

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(PLUGIN_DIR, "scripts", "mcp_server.py")
KINDS = ("text", "image", "audio", "video")

_proc = None  # long-lived stdio-server child (engine venv); spawned on demand


def _root() -> str:  # pragma: no cover - exercised inside the child's tool layer
    cfg = os.path.join(PLUGIN_DIR, "mediarag-config.yaml")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                if line.startswith("root:"):
                    return line.split(":", 1)[1].strip().strip('"')
    return os.environ.get("MEDIARAG_ROOT") or os.path.expanduser("~/.mediarag")


@functools.lru_cache(maxsize=1)
def _engine_python() -> str | None:
    """Newest installs-env venv whose python can import the engine (hashes rotate).

    installs/ may live under the shared Hermes home (~/.hermes) or under a
    profile-scoped HERMES_HOME; probe whichever exist, newest first.
    """
    roots = []
    try:
        from hermes_constants import get_hermes_home
        roots.append(os.path.join(get_hermes_home(), "installs"))
    except ImportError:
        pass  # standalone import (tests, editor) — expanduser root still probed
    roots.append(os.path.join(os.path.expanduser("~/.hermes"), "installs"))
    cands = []
    for root in dict.fromkeys(roots):  # dedupe, keep order
        cands += glob.glob(os.path.join(
            root, "*", "environments", "*", "venv", "bin", "python"))
    cands.sort(key=os.path.getmtime, reverse=True)
    for py in cands:
        try:
            r = subprocess.run(
                # matches _engine_importable(): the processor chain needs BOTH
                # torch and torchvision or every embed raises at call time
                [py, "-c",
                 "import transformers, onnxruntime, numpy, torch, torchvision"],
                capture_output=True, timeout=120,
            )
            if r.returncode == 0:
                return py
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def _spawn():
    global _proc
    py = _engine_python()
    if py is None:
        raise HTTPException(
            status_code=503,
            detail="MediaRAG engine deps not found in any installs env — "
                   "install the engine venv (pip install -r requirements.txt)",
        )
    _proc = subprocess.Popen(
        [py, SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, cwd=PLUGIN_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _rpc("initialize", {}, rid=0)
    _rpc("notifications/initialized", {}, rid=None)


def _rpc(method: str, params: dict, rid: int | None = 1) -> dict | None:
    assert _proc is not None
    msg = {"jsonrpc": "2.0", "method": method}
    if rid is not None:
        msg["id"] = rid
    msg["params"] = params or {}
    _proc.stdin.write(json.dumps(msg) + "\n")
    _proc.stdin.flush()
    if rid is None:
        return None
    while True:
        line = _proc.stdout.readline()
        if not line:
            raise RuntimeError("mediarag stdio server exited")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            continue
        if resp.get("id") == rid:
            if "error" in resp:
                raise RuntimeError(str(resp["error"].get("message", resp["error"])))
            return resp.get("result")


def _call(tool: str, args: dict) -> dict:
    global _proc
    if _proc is None or _proc.poll() is not None:
        _spawn()  # (re)start: engine venv python + stdio server, warm model
    result = _rpc("tools/call", {"name": tool, "arguments": args})
    text = (result or {}).get("content", [{}])[0].get("text", "{}")
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = {"raw": text}
    if (isinstance(out, dict) and out.get("isError")) or str(out.get("raw", "")).startswith("error:"):
        raise RuntimeError(str(out.get("raw") or out))
    return out


@router.get("/status")
async def status():
    return await asyncio.to_thread(lambda: _call("mediarag_status", {}))


@router.get("/search")
async def search(q: str, kind: str | None = None, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    if kind and kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {KINDS}")
    k = min(max(k, 1), 20)
    return await asyncio.to_thread(
        lambda: _call("mediarag_search",
                      {"query": q.strip(), "k": k, "include_thumbnails": True,
                       "kind": kind or None}))
