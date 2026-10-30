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
import json
import os
import re
import subprocess
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
    """Import the plugin as a package (its register() uses relative imports)."""
    parent = os.path.dirname(REPO)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import mediarag  # noqa: E402  — the package __init__ under test
    return mediarag


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

    # ---- the gateway process is left alone --------------------------------
    check("no sys.path helper survives", not hasattr(init, "_prepare_env"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    before = list(sys.path)

    class RegCtx:
        def __init__(self):
            self.calls = []

        def register_tool(self, **kw):
            self.calls.append(kw)

    ctx = RegCtx()
    init.register(ctx)
    check("register() does not set HF_HUB_OFFLINE process-wide",
          "HF_HUB_OFFLINE" not in os.environ)
    check("register() does not touch sys.path", sys.path == before)
    check("no tool schema exposes a root argument",
          all("root" not in kw["schema"]["parameters"]["properties"] for kw in ctx.calls),
          str([list(kw["schema"]["parameters"]["properties"]) for kw in ctx.calls]))
    check("all five tools registered", len(ctx.calls) == 5, str(len(ctx.calls)))

    # ---- the two reviewer 'recommended' items ----------------------------
    fm = (REPO and open(os.path.join(REPO, "scripts", "fetch_model.py")).read()) or ""
    check("fetch_model pins revision=", "revision=" in fm)
    eng = open(os.path.join(REPO, "core", "engine.py")).read()
    check("from_pretrained asks for local files only", "local_files_only=True" in eng)

    rclone = os.path.join(REPO, "adapters", "rclone")
    bad = subprocess.run([sys.executable, rclone, "fetch", "../escape:x"],
                         capture_output=True, text=True)
    check("rclone adapter refuses a path-traversal remote",
          bad.returncode != 0 and "invalid remote" in (bad.stderr + bad.stdout), bad.stderr[:120])

    # ---- adapters must not carry a narrower extension list than the indexer ---
    # They used to emit only md/txt/subs/images/audio, so a Dropbox or pCloud
    # folder of pdfs/epubs indexed to nothing, silently. The indexer classifies;
    # adapters emit and let it answer `unsupported_ext`.
    for adapter, env_var in (("dropbox", "DROPBOX_DIR"), ("pcloud", "PCLOUD_MOUNT")):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "Papers"))
            for name in ("a.pdf", "b.epub", "c.docx", "d.md", "e.sh"):
                open(os.path.join(td, "Papers", name), "wb").write(b"x")
            got = subprocess.run([sys.executable, os.path.join(REPO, "adapters", adapter), "fetch", "Papers"],
                                 capture_output=True, text=True, env={**os.environ, env_var: td})
        emitted = got.stdout
        docs = [n for n in ("a.pdf", "b.epub", "c.docx") if f"Papers/{n}" in emitted]
        check(f"{adapter} adapter emits document formats the indexer supports",
              len(docs) == 3, f"{got.returncode} {docs} {got.stderr[:100]}")
        # A URI path is "/"-separated on every platform. os.path.relpath hands back
        # "Papers\a.pdf" on Windows, which made these adapters emit malformed URIs
        # (and the assertion above fail on the windows CI lane only).
        check(f"{adapter} adapter URIs are '/'-separated",
              "\\" not in emitted and emitted.strip() != "",
              emitted[:120])

    # ---- MCP entry must survive a client that has no engine installed ------
    # `command: python3` on a stranger's machine is the system python; the server
    # died mid-handshake and the client only saw "Connection closed". The entry
    # now runs a stdlib-only launcher that finds a provisioned interpreter.
    mcp_json = json.load(open(os.path.join(REPO, "mcp.json")))
    entry = mcp_json["mcpServers"]["mediarag"]
    # Spec: "A plugin that bundles an executable in the package MUST use a
    # plugin-relative command" and portable plugins must not depend on PATH.
    check("mcp.json command is plugin-relative, not a bare python3",
          entry["command"].startswith("./") and entry["command"].endswith("mcp_launch.py"),
          entry["command"])

    launcher_path = os.path.join(REPO, "scripts", "mcp_launch.py")
    launcher = open(launcher_path).read()
    needs = ("numpy", "onnxruntime", "transformers", "torch")
    check("launcher probes the whole engine stack",
          all(n in launcher.split("PROBE = ")[1].split("\n")[0] for n in needs), "probe incomplete")
    spec = importlib.util.spec_from_file_location("mcp_launch", launcher_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check("launcher rejects a missing interpreter without raising",
          mod._usable("/nonexistent/python3") is False)

    # ---- adapters must not need an exec bit or a shebang -------------------
    # Wheels and Windows checkouts carry no exec bit; running the file directly
    # was a PermissionError waiting for the first user sources.d adapter.
    body = open(os.path.join(REPO, "core", "tools.py")).read() \
        .split("def run_adapter_sink")[1].split("\ndef ")[0]
    check("adapters run via sys.executable on every platform",
          '[sys.executable, adapter, "fetch", rest]' in body and 'os.name == "nt"' not in body,
          "exec path branches on platform again")
    check("adapter spawn failure returns adapter_error, not an exception",
          "except OSError" in body)

    # ---- one version, four files -----------------------------------------
    def _ver(path, pattern):
        m = re.search(pattern, open(os.path.join(REPO, path)).read(), re.M)
        return m.group(1) if m else None

    versions = {
        "plugin.yaml": _ver("plugin.yaml", r"^version:\s*(\S+)"),
        "plugin.json": _ver("plugin.json", r'"version":\s*"([^"]+)"'),
        "mcp_server": _ver("scripts/mcp_server.py", r'"version":\s*"([^"]+)"'),
        "pyproject": _ver("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
    }
    check(f"version consistent across manifests: {versions}",
          None not in versions.values() and len(set(versions.values())) == 1)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
