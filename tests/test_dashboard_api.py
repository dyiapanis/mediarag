"""MediaRAG dashboard backend: plugin_api routes (offline) + one real-child roundtrip.

Run: python tests/test_dashboard_api.py   (skips when fastapi is absent)
"""
import asyncio
import json
import os
import sys

try:
    from fastapi import HTTPException
except ImportError:  # CI lanes without the gateway's fastapi dep
    print("SKIP test_dashboard_api: fastapi not installed")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.machinery
import importlib.util

os.environ.setdefault("MEDIARAG_ROOT", "/tmp/phoenix-dashboard-test-corpus")
spec = importlib.util.spec_from_file_location(
    "mediarag_plugin_api", os.path.join(ROOT, "dashboard", "plugin_api.py"),
    loader=importlib.machinery.SourceFileLoader("mediarag_plugin_api", os.path.join(ROOT, "dashboard", "plugin_api.py")))
assert spec is not None and spec.loader is not None
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

import unittest.mock as um  # noqa: E402


def check(name, cond, detail=None):
    if not cond:
        print("FAIL", name, ":", detail)
        raise SystemExit(1)
    print("PASS", name)


FAKE_RESULTS = {
    "results": [
        {"uri": "/x/banner.png", "kind": "image", "score": 0.82, "span_start": 0.0,
         "span_end": 0.0, "snippet": "banner", "thumbnail": "data:image/png;base64,QQ=="},
        {"uri": "/x/clip.wav", "kind": "audio", "score": 0.71, "span_start": 4.0,
         "span_end": 15.2, "snippet": "clip [4.0-15.2s]", "thumbnail": None},
    ]
}


def rpc_reply(rid, payload):
    """The stdio server's reply frame for a successful tools/call."""
    return json.dumps({"jsonrpc": "2.0", "id": rid,
                       "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})


class FakeProc:
    """Scripted newline-JSON-RPC child."""

    def __init__(self, replies):
        self.lines = list(replies)
        self.sent = []

        class _In:
            def __init__(self, host):
                self._h = host

            def write(self, s):
                self._h.sent.append(s)

            def flush(self):
                pass

        self.stdin = _In(self)

        class _Out:
            def __init__(self, host):
                self._h = host

            def readline(self):
                return self._h.lines.pop(0) + "\n"

        self.stdout = _Out(self)

    def poll(self):
        return None


def with_child(replies, fn=None):
    fake = FakeProc(replies)
    sentinel_api = api
    with um.patch.object(sentinel_api, "_proc", fake), \
         um.patch.object(sentinel_api, "_spawn", lambda: None):
        return fn() if fn else None


# --- manifest sanity ----------------------------------------------------------
with open(os.path.join(ROOT, "dashboard", "manifest.json")) as f:
    manifest = json.load(f)
check("manifest name", manifest["name"] == "mediarag")
check("entry exists", os.path.exists(os.path.join(ROOT, "dashboard", manifest["entry"])))

# --- rpc: rid-matched reply ----------------------------------------------------
def _probe1():
    out = api._rpc("tools/call", {"name": "mediarag_x"}, rid=37)
    check("rpc returns rid-matched result",
          out == {"content": [{"type": "text", "text": json.dumps(FAKE_RESULTS)}]}, out)
    check("rpc wrote the request", "tools/call" in (api._proc.sent[0] if api._proc.sent else ""), api._proc.sent)

replies = [rpc_reply(37, FAKE_RESULTS)]
with_child(replies, _probe1)

# --- _call passthrough ---------------------------------------------------------
def _probe2():
    out = api._call("mediarag_search", {"query": "quokka"})
    check("call passthrough", out == FAKE_RESULTS, out)

replies = [rpc_reply(1, FAKE_RESULTS)]
with_child(replies, _probe2)

# child-reported tool error surfaces as a raised error
def _probe3():
    try:
        api._call("mediarag_search", {})
        check("child tool error raises", False, "no exception")
    except RuntimeError as e:
        check("child tool error raises", "boom" in str(e), str(e))

err_reply = [json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
    {"type": "text", "text": "error: boom"}], "isError": True}})]
with_child(err_reply, _probe3)

# dead child triggers a respawn attempt (503 offline is a clean refusal)
def _probe4():
    try:
        api._call("mediarag_status", {})
    except RuntimeError:
        pass
    except HTTPException:
        pass

respawned = {"n": 0}
fake = FakeProc([rpc_reply(1, {"ok": 1})])
fake.poll = lambda: 1  # type: ignore[method-assign]  # pretend the child died
with um.patch.object(api, "_proc", fake), \
     um.patch.object(api, "_spawn", (lambda: respawned.__setitem__("n", respawned["n"] + 1))):
    api._call("mediarag_status", {})
check("dead child triggers respawn attempt", respawned["n"] == 1, respawned)

# --- handlers (via stubbed _call) ----------------------------------------------
with um.patch.object(api, "_call", lambda t, a: {"results": FAKE_RESULTS["results"]}
                     if t == "mediarag_search" else {"total": 2}):
    hits = asyncio.run(api.search("quokka lantern", k=8))
    check("search passthrough", hits == {"results": FAKE_RESULTS["results"]}, hits)
    st = asyncio.run(api.status())
    check("status passthrough", st == {"total": 2}, st)

# --- validation unchanged -----------------------------------------------------
try:
    asyncio.run(api.search(""))
    check("empty query rejected", False, "no exception")
except HTTPException as e:
    check("empty query rejected", e.status_code == 400, e.status_code)

try:
    asyncio.run(api.search("x", kind="warp_drive"))
    check("bad kind rejected", False, "no exception")
except HTTPException as e:
    check("bad kind rejected", e.status_code == 400, e.status_code)

vres = asyncio.run(api.search("frame", kind="video"))
check("kind=video accepted", "results" in vres, vres)

# --- 503 when no engine env resolves ------------------------------------------
with um.patch.object(api, "_engine_python", lambda: None), \
     um.patch.object(api, "_proc", None):
    api._proc = None
    try:
        asyncio.run(api.status())
        check("no-engine -> 503", False, "no exception")
    except HTTPException as e:
        check("no-engine -> 503", e.status_code == 503, e.status_code)

# --- index.js parses ----------------------------------------------------------
import shutil
import subprocess
if shutil.which("node"):
    # list-argv, no shell: the path is os.path.join'd, nothing user-supplied
    r = subprocess.run(["node", "--check", os.path.join(ROOT, "dashboard", "dist", "index.js")],
                       capture_output=True)
    check("index.js parses", r.returncode == 0, r.stderr.decode()[:200])
else:
    print("note: node absent — index.js not syntax-checked")

# --- real child roundtrip (local only; CI has no installs env) ----------------
py = api._engine_python()
if py:
    api._proc = None
    st = asyncio.run(api.status())
    check("REAL child roundtrip: status", isinstance(st, dict) and "total" in st,
          str(st)[:120])
else:
    print("SKIP real-child roundtrip: no engine env on this box")

print("\nALL DASHBOARD API CHECKS PASS")
