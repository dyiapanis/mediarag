#!/usr/bin/env python3
"""Contract regressions for the v0.1.4 hardening batch. No model, no network.

Direct-run: python3 tests/test_contract.py (exit 0 = all pass).

Covers the defects the audit found: the unchanged sentinel leaking into tool
output, skip counts disagreeing with skip lists, adapter stderr filed as skips,
adapter discovery reaching $HOME, Windows path separators defeating the exclude
patterns, the engine probe passing without torch/torchvision, and the two-phase
KNN / joined keyword lanes returning what brute force says they should.
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np

try:  # pre-bind venv copies before engine prepends ~/pylibs2 (fleet env only)
    import regex  # noqa: F401
    import PIL  # noqa: F401
except ImportError:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core import ingest, store, tools  # noqa: E402

PASSED = FAILED = 0
DIM = 64  # embeddings are swapped for fakes here; dimension is irrelevant


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}  {detail}")


def fake_vec(_text: str) -> np.ndarray:
    return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


def _plugin_init():
    """Load the plugin __init__ as a module without calling register()."""
    spec = importlib.util.spec_from_file_location(
        "mr_plugin_init", os.path.join(REPO, "__init__.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mediarag-contract-")
    root = os.path.join(tmp, "corpus")
    docs = os.path.join(tmp, "docs")
    os.makedirs(docs)

    # ---- exclude patterns: relative to the walk root, both separators -----
    check("exclude .obsidian (posix)", ingest.excluded("/srv/Vault/.obsidian/app.json", "/srv/Vault"))
    check("exclude cache (posix)", ingest.excluded("/srv/Media/cache/shot.png", "/srv/Media"))
    check("exclude .obsidian (windows)", ingest.excluded(r"C:\Vault\.obsidian\app.json", "C:/Vault"))
    check("exclude cache (windows)", ingest.excluded(r"C:\Media\cache\shot.png", "C:/Media"))
    check("keep ordinary file (windows)", not ingest.excluded(r"C:\Media\notes\a.md", "C:/Media"))
    check("root-level .DS_Store excluded", ingest.excluded("/srv/Media/.DS_Store", "/srv/Media"))
    check("a root under .../cache/ still indexes",
          not ingest.excluded("/srv/Media/cache/deep/note.md", "/srv/Media/cache"))

    # ---- adapter discovery: exactly two dirs, $HOME not among them --------
    dirs = tools._adapter_dirs("/tmp/corpus")
    check("adapter dirs == 2", len(dirs) == 2, str(dirs))
    check("no $HOME/adapters in scan",
          not any(d == os.path.join(os.path.expanduser("~"), "adapters") for d in dirs), str(dirs))
    check("shipped adapters resolve",
          {"obsidian", "pcloud", "raindrop", "rclone"} <= set(tools.list_adapters("/tmp/corpus")))

    # ---- tool contract: unchanged file is not a negative add --------------
    ingest.embed_doc = fake_vec  # module-level import, so this is the live symbol
    body = "# Title\n\n" + ("lorem ipsum dolor sit amet consectetur " * 20)
    one = os.path.join(docs, "one.md")
    with open(one, "w", encoding="utf-8") as fh:
        fh.write(body)
    first = tools.tool_add_source(one, root)
    second = tools.tool_add_source(one, root)
    check("first add indexes rows", first["added"] > 0, str(first))
    check("re-add reports added == 0 (no -1 sentinel)", second["added"] == 0, str(second))

    # ---- skip counts and skip lists agree --------------------------------
    many = os.path.join(tmp, "many")
    os.makedirs(many)
    for i in range(25):
        with open(os.path.join(many, f"tiny{i}.md"), "w", encoding="utf-8") as fh:
            fh.write("x")  # below MIN_CONTENT_CHARS -> "empty" skip
    c = store.connect(root)
    try:
        rep = ingest.index_dir(c, many)
        check("skips_total counts every skip", rep["skips_total"] == 25, str(rep))
        check("skipped count matches total", rep["skipped"] == rep["skips_total"], str(rep))
        check("detail list is capped at 20", len(rep["skipped_detail"]) == 20, str(len(rep["skipped_detail"])))
        check("truncation is declared", rep["skips_truncated"] is True, str(rep))
        check("each entry carries a code", all("code" in s for s in rep["skipped_detail"]))

        # ---- engine failure has its own code, not read_error --------------
        def boom(_text):
            raise ImportError("EmbeddingGemma2VideoProcessor requires the PyTorch library")
        ingest.embed_doc = boom
        bad = os.path.join(docs, "engine.md")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(body)
        n, skip = ingest.index_file(c, bad)
        check("engine failure -> engine_unavailable",
              n == 0 and skip and skip[0]["code"] == "engine_unavailable", str(skip))
        ingest.embed_doc = fake_vec

        # ---- store pragmas -------------------------------------------------
        check("journal_mode is wal",
              c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal")
        check("busy_timeout set",
              c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000)

        # ---- two-phase KNN == brute force (fresh corpus: notes only) ------
        rng = np.random.default_rng(0)
        kc = store.connect(os.path.join(tmp, "knn"))
        vectors = {}
        for i in range(30):
            v = rng.random(DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            uri = f"note://t/{i}"
            vectors[uri] = v
            store.upsert_uri(kc, uri, [{"uri": uri, "kind": "text", "span_start": 0.0,
                                        "span_end": 0.0, "snippet": f"doc {i} lorem",
                                        "vec": v, "thumb": None}])
        q = rng.random(DIM).astype(np.float32)
        q /= np.linalg.norm(q)
        hits = store.search(kc, q, k=5)
        expect = sorted(vectors, key=lambda u: -float(vectors[u] @ q))[:5]
        check("search order matches brute force", [h["uri"] for h in hits] == expect,
              f"{[h['uri'] for h in hits]} vs {expect}")
        check("scores descending",
              all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1)))
        check("no thumbnails when off",
              all(h["thumbnail"] is None for h in store.search(kc, q, k=5, include_thumbnails=False)))

        # ---- joined keyword lane ------------------------------------------
        kw = store.keyword_search(kc, "lorem", k=5)
        check("keyword lane returns joined rows", len(kw) == 5 and all(h["uri"] for h in kw), str(len(kw)))
        check("keyword lane hydrates snippet/kind",
              all(h["snippet"] and h["kind"] == "text" for h in kw))
        check("keyword kind filter narrows",
              store.keyword_search(kc, "lorem", k=5, kind="image") == [])
        no_thumb = store.keyword_search(kc, "lorem", k=5, include_thumbnails=False)
        check("keyword lane honours include_thumbnails",
              all(h["thumbnail"] is None for h in no_thumb))
        kc.close()
    finally:
        c.close()

    # ---- add_source URL policy: every hop checked, curl never follows ------
    import http.server
    import threading
    from core import urlsafe
    ingest.cache_dir = lambda: tmp
    os.environ.pop("MEDIARAG_ALLOW_PRIVATE_URLS", None)
    urlsafe._host_is_safe_url = lambda: None  # standalone rules
    real_run = tools.subprocess.run
    calls = []
    tools.subprocess.run = lambda *a, **k: calls.append(a) or real_run(*a, **k)
    for bad in ("http://127.0.0.1/", "http://10.0.0.5/", "http://169.254.10.10/"):
        got = tools.fetch_url(bad)
        check(f"refuses {bad}", got["code"] == "blocked_url", str(got))
    check("refused URLs never reach curl", calls == [], str(calls))

    class Hops(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/page.txt":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"hello")
                return
            self.send_response(302)
            self.send_header("Location", "/page.txt" if self.path == "/ok"
                             else "http://169.254.10.10/x")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hops)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    os.environ["MEDIARAG_ALLOW_PRIVATE_URLS"] = "1"
    try:
        got = tools.fetch_url(base + "/ok")
        check("same-host redirect followed", got["code"] == "ok" and got["size"] == 5, str(got))
        got = tools.fetch_url(base + "/away")
        check("redirect to link-local refused", got["code"] == "blocked_url", str(got))
    finally:
        os.environ.pop("MEDIARAG_ALLOW_PRIVATE_URLS", None)
        srv.shutdown()
        tools.subprocess.run = real_run

    # ---- a per-call root never selects which adapters run ----------------
    other = os.path.join(tmp, "other-root")
    os.makedirs(os.path.join(other, "sources.d"))
    probe = os.path.join(other, "sources.d", "probe")
    with open(probe, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nexit 0\n")
    os.chmod(probe, 0o755)
    got = tools.tool_add_source("probe://x", other)
    check("per-call root adapters are not loaded", got.get("code") == "unknown_scheme", str(got))
    check("status lists configured adapters only",
          "probe" not in tools.tool_status(other)["source_adapters"])

    # ---- engine probe must notice a missing torch -------------------------
    init = _plugin_init()
    saved = {k: sys.modules.get(k) for k in ("torch", "torchvision")}
    try:
        sys.modules["torch"] = None  # any import raises ImportError
        sys.modules["torchvision"] = None
        check("probe catches missing torch", init._engine_importable() is False)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
