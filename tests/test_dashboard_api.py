"""MediaRAG dashboard backend: plugin_api routes (offline, real corpus untouched).

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

# Import the repo copy (dashboard/plugin_api.py), not any deployed copy.
os.environ["MEDIARAG_ROOT"] = "/tmp/phoenix-dashboard-test-corpus"
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


class FakeTools:
    calls = []

    @staticmethod
    def tool_search(q, k=5, kind=None, include_thumbnails=True, root=None):
        FakeTools.calls.append(("search", q, k, kind, root))
        return dict(FAKE_RESULTS)

    @staticmethod
    def tool_status(root=None, **_x):
        FakeTools.calls.append(("status", root))
        return {"total": 2, "counts": {"image": 1, "audio": 1}, "uris": 2, "root": root}


# --- manifest sanity ----------------------------------------------------------
with open(os.path.join(ROOT, "dashboard", "manifest.json")) as f:
    manifest = json.load(f)
check("manifest name", manifest["name"] == "mediarag")
check("icon is mapped", manifest["icon"] in {
    "Activity", "BarChart3", "Clock", "Code", "Database", "Eye", "FileText", "Globe",
    "Heart", "KeyRound", "MessageSquare", "Package", "Puzzle", "Settings", "Shield",
    "Sparkles", "Star", "Terminal", "Wrench", "Zap"}, manifest["icon"])
check("entry exists", os.path.exists(os.path.join(ROOT, "dashboard", manifest["entry"])))

# --- handlers: passthrough + validation ---------------------------------------
api._tools = lambda: FakeTools  # stub the deployed-package import

hits = asyncio.run(api.search("quokka lantern", k=8))
check("search passthrough", hits == FAKE_RESULTS, hits)
check("search args", FakeTools.calls[-1] == ("search", "quokka lantern", 8, None,
                                             "/tmp/phoenix-dashboard-test-corpus"), FakeTools.calls[-1])

asyncio.run(api.search("again", kind="audio", k=99))
check("kind passthrough + k clamped", FakeTools.calls[-1] == ("search", "again", 20, "audio",
                                                              "/tmp/phoenix-dashboard-test-corpus"), FakeTools.calls[-1])

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

st = asyncio.run(api.status())
check("status passthrough + root wired", st["total"] == 2 and st["root"] == "/tmp/phoenix-dashboard-test-corpus", st)

vres = asyncio.run(api.search("frame", kind="video"))
check("kind=video accepted", vres == FAKE_RESULTS and FakeTools.calls[-1][:4] == ("search", "frame", 5, "video"), FakeTools.calls[-1])

# --- index.js is syntactically valid JS (node when available) ------------------
import shutil
import subprocess
if shutil.which("node"):
    # list-argv, no shell: the path is os.path.join'd, nothing user-supplied
    r = subprocess.run(["node", "--check", os.path.join(ROOT, "dashboard", "dist", "index.js")],
                       capture_output=True)
    check("index.js parses", r.returncode == 0, r.stderr.decode()[:200])
else:
    print("note: node absent — index.js not syntax-checked")

print("\nALL DASHBOARD API CHECKS PASS")
