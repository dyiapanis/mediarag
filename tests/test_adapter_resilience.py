"""Adapter resilience: timeout salvage and fail-closed secret handling.

Run: python tests/test_adapter_resilience.py   (subprocess + embeddings stubbed)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core import engine, store, tools  # noqa: E402


def check(name, cond, detail=None):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


root = tempfile.mkdtemp(prefix="mediarag-adapter-")
real_run = tools.subprocess.run
real_embed = engine.embed_doc
real_root = tools.DEFAULT_ROOT
engine.embed_doc = lambda text: [0.0] * 8
tools.DEFAULT_ROOT = root
try:
    os.makedirs(os.path.join(root, "sources.d"))
    for scheme in ("fake", "notion"):
        with open(os.path.join(root, "sources.d", scheme), "w") as fh:
            fh.write("# stub adapter; subprocess is faked in this test\n")

    lines = [
        {"uri": "fake://lib/a", "title": "a", "text": "marmalade lanterns " * 20},
        {"uri": "fake://lib/b", "title": "b", "text": "second document body " * 20},
        "not json at all",
    ]
    payload = ("\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines)).encode()

    def timeout_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1800, output=payload, stderr=b"syncing 2 of 900")

    tools.subprocess.run = timeout_run
    res = tools.run_adapter_sink("fake://lib", root)
    check("timeout is reported, not raised", res.get("code") == "adapter_timeout", res)
    check("timeout is marked partial", res.get("partial") is True, res)
    check("complete items were ingested", res["added"] > 0, res)
    c = store.connect(root)
    n = c.execute("SELECT COUNT(*) FROM assets WHERE uri LIKE 'fake://%'").fetchone()[0]
    check("rows landed in the store", n == res["added"], (n, res["added"]))
    c.close()
    check("adapter stderr rides along", any("syncing" in ln for ln in res["adapter_log"]), res)

    # A NON-ZERO EXIT gets the same salvage: complete items are kept and the
    # failure is reported, instead of every emitted item being thrown away.
    # Distinct URIs + exact counts, so this cannot pass on the timeout case's rows.
    exit_payload = b"\n".join([
        json.dumps({"uri": "fake://exit/a", "title": "a", "text": "marmalade lanterns " * 20}).encode(),
        json.dumps({"uri": "fake://exit/b", "title": "b", "text": "second document body " * 20}).encode(),
        b"not json at all",
    ])

    def failing_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout=exit_payload, stderr=b"rate limited at item 900")

    tools.subprocess.run = failing_run
    res = tools.run_adapter_sink("fake://exit", root)
    check("non-zero exit is reported", res.get("code") == "adapter_error", res)
    check("non-zero exit is marked partial", res.get("partial") is True, res)
    check("non-zero exit still ingests complete items", res["added"] == 2, res)
    check("non-zero exit keeps the stderr tail", "rate limited" in res.get("stderr", ""), res)
    c = store.connect(root)
    n = c.execute("SELECT COUNT(*) FROM assets WHERE uri LIKE 'fake://exit/%'").fetchone()[0]
    check("non-zero exit rows persisted (exact)", n == 2 and n == res["added"], (n, res["added"]))
    c.close()

    # A failing secret scope must fail closed with a clear error, not run the
    # adapter unauthenticated — and must not affect schemes without a token.
    agent = types.ModuleType("agent")
    scope = types.ModuleType("agent.secret_scope")

    def broken(name):
        raise RuntimeError("pass store locked")

    scope.get_secret = broken
    agent.secret_scope = scope
    sys.modules["agent"] = agent
    sys.modules["agent.secret_scope"] = scope
    called = []

    def should_not_run(cmd, **kw):
        called.append(cmd)
        raise AssertionError("adapter must not run when the scope fails")

    tools.subprocess.run = should_not_run
    res = tools.run_adapter_sink("notion://abc", root)
    check("scope failure fails closed", res.get("code") == "secret_error", res)
    check("adapter never spawned", called == [], called)
    check("error names the failure", "pass store locked" in res.get("detail", ""), res)
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.secret_scope", None)
finally:
    tools.subprocess.run = real_run
    engine.embed_doc = real_embed
    tools.DEFAULT_ROOT = real_root
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.secret_scope", None)
    shutil.rmtree(root, ignore_errors=True)

print("\nALL ADAPTER RESILIENCE CHECKS PASS")
