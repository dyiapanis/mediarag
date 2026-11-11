"""Text-adapter sink fidelity: body text must be chunked, stored and searchable.

Regression for the review finding that run_adapter_sink embedded the whole body
as one row (1024-token truncation) and stored only the title as the snippet,
leaving nothing for the keyword lane or the result view.

Run: python tests/test_adapter_sink_text.py   (model stubbed; no network)
"""
import os
import shutil
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import store, tools  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


tmp = tempfile.mkdtemp(prefix="mediarag-sink-")
try:
    os.makedirs(os.path.join(tmp, "sources.d"), exist_ok=True)
    body = "# Project\n\n" + ("filler paragraph text. " * 80) + "\n\n" + \
           "The quokka migration finished on Friday.\n\n" + ("more filler text. " * 60)
    with open(os.path.join(tmp, "sources.d", "fake"), "w") as fh:
        fh.write("#!/usr/bin/env python3\nimport json,sys\n"
                 f"print(json.dumps({{'uri':'fake://project','title':'Project','text':{body!r}}}))\n")

    # Stub the engine module so no model is needed; sink imports it at call time.
    fake = types.ModuleType("core.engine")
    fake.embed_doc = lambda t: np.ones(4, dtype=np.float32)
    sys.modules["core.engine"] = fake

    # Adapters load from the configured corpus root only (never the per-call
    # root), so the fake must be discoverable through DEFAULT_ROOT.
    tools.DEFAULT_ROOT = tmp
    report = tools.run_adapter_sink("fake://project", tmp)
    check("multiple chunks stored", report["added"] >= 2, report)

    c = store.connect(tmp)
    rows = c.execute("SELECT snippet FROM assets WHERE uri='fake://project'").fetchall()
    check("body present in stored rows", any("quokka migration" in (s or "") for (s,) in rows),
          [s[:60] for (s,) in rows])
    check("no title-only rows", all(len(s) > 20 for (s,) in rows), [s[:40] for (s,) in rows])
    kw = c.execute("SELECT uri FROM kw_fts WHERE text LIKE '%quokka%'").fetchall()
    check("keyword lane indexes the body", kw, kw)
    c.close()
finally:
    sys.modules.pop("core.engine", None)
    shutil.rmtree(tmp, ignore_errors=True)

print("\nALL ADAPTER SINK TEXT CHECKS PASS")
