"""MediaRAG tool surface: the five MCP/plugin tools."""
from __future__ import annotations

import os
import subprocess

from . import ingest as ing
from . import store
from . import search as se

DEFAULT_ROOT = os.path.expanduser("~/.mediarag")


def _root(root: str | None) -> str:
    return root or DEFAULT_ROOT


def tool_search(query: str, k: int = 5, root: str | None = None,
                include_thumbnails: bool = True, kind: str | None = None,
                agent: str | None = None, **_x) -> dict:
    return {"results": se.search(_root(root), query, k=k, kind=kind,
                                 include_thumbnails=include_thumbnails, agent=agent)}


def tool_ingest_dir(path: str, root: str | None = None, **_x) -> dict:
    c = store.connect(_root(root))
    try:
        return ing.index_dir(c, path)
    finally:
        c.close()


def tool_add_source(source: str, root: str | None = None, **_x) -> dict:
    """File, dir, http(s) URL, or adapter scheme URI (e.g. raindrop://…/pcloud://…)."""
    root = _root(root)
    if "://" in source and not source.startswith(("http://", "https://")):
        return run_adapter_sink(source, root)
    if source.startswith(("http://", "https://")):
        fetched = fetch_url(source)
        if fetched["code"] != "ok":
            return fetched
        p = fetched["path"]
    else:
        p = os.path.expanduser(source)
        if p.endswith("/"):
            p = p.rstrip("/")
        if not os.path.exists(p):
            return {"code": "not_found", "source": source}
    c = store.connect(root)
    try:
        if os.path.isdir(p):
            return ing.index_dir(c, p)
        added, skipped = ing.index_file(c, p)
        return {"added": added, "skipped": skipped}
    finally:
        c.close()


def tool_add_note(text: str, root: str | None = None, **_x) -> dict:
    return se.add_note(_root(root), text)


def tool_status(root: str | None = None, **_x) -> dict:
    root = _root(root)
    c = store.connect(root)
    try:
        s = store.stats(c)
    finally:
        c.close()
    s["root"] = root
    s["source_adapters"] = list_adapters(root)
    return s


# --- add_source lanes -------------------------------------------------------

_UA = {"User-Agent": "MediaRAG/1.0"}


def fetch_url(url: str) -> dict:
    """Download (≤256MB, 120s) to the persistent cache; return {'code','path'}."""
    import shutil
    if not shutil.which("curl"):
        raise RuntimeError("curl not installed")
    out = os.path.join(ing.cache_dir(), "_url", _fn_from_url(url))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run(
        ["curl", "-fsSL", "--proto", "=http,https", "--proto-redir",
         "=http,https", "--max-time", "120", "-o", out, url],
        capture_output=True, timeout=130,
    )
    if r.returncode != 0:
        return {"code": "fetch_error", "url": url, "detail": r.stderr.decode()[-200:]}
    size = os.path.getsize(out)
    if size > 256 * 1024 * 1024:
        os.unlink(out)
        return {"code": "too_large", "url": url, "size": size}
    return {"code": "ok", "path": out, "size": size}


def _fn_from_url(url: str) -> str:
    import hashlib
    import re
    tail = os.path.basename(url.split("?")[0]) or "bin"
    tail = re.sub(r"[^A-Za-z0-9._-]", "_", tail)[:64] or "bin"
    if "." not in tail:
        tail += ".bin"
    return f"{hashlib.sha1(url.encode()).hexdigest()[:10]}-{tail}"


def run_adapter_sink(source: str, root: str) -> dict:
    """Adapter scheme URI → runner JSON-lines contract (sources.d)."""
    scheme, rest = source.split("://", 1)
    adapters = list_adapters(root)
    adapter = adapters.get(scheme)
    if not adapter:
        return {"code": "unknown_scheme", "scheme": scheme,
                "available": sorted(adapters)}
    # stdout JSON lines: {"uri","title","text"} text items | {"uri","path"} file items
    r = subprocess.run([adapter, "fetch", rest], capture_output=True, timeout=1800)
    if r.returncode != 0:
        return {"code": "adapter_error", "scheme": scheme, "stderr": r.stderr.decode()[-300:]}
    from .engine import embed_doc
    import json as _json
    added = 0
    skipped = []
    c = store.connect(root)
    try:
        for line in r.stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = _json.loads(line)
            except ValueError:
                continue
            if "text" in item:
                v = embed_doc(item["text"])
                row = {"uri": item["uri"], "kind": "text", "span_start": 0.0, "span_end": 0.0,
                       "snippet": item.get("title", "") or item["text"][:200], "vec": v, "thumb": None}
                store.upsert_uri(c, item["uri"], [row])
                added += 1
            elif "path" in item and os.path.exists(item["path"]):
                n, skip = ing.index_file(c, item["path"])
                added += n
                for s in skip:
                    s["uri"] = item["uri"]
                    skipped.append(s)
    finally:
        c.close()
    if r.stderr:
        for line in r.stderr.decode().splitlines():
            if line.strip():
                skipped.append({"code": "adapter_skip", "detail": line.strip()[:200]})
    return {"added": added, "skipped": skipped}


def list_adapters(root: str) -> dict:
    """Shipped ./adapters/ executables + <root>/sources.d/<scheme> user ones.

    Walks up from this module to find the package's adapters/ dir (works from
    the repo layout src/mediarag as well as a deployed plugin dir layout).
    """
    adapters: dict[str, str] = {}
    dirs = [os.path.join(root, "sources.d")] if root else []
    base = os.path.dirname(os.path.abspath(__file__))  # .../src/mediarag (or deployed pkg)
    for _ in range(3):
        dirs.append(os.path.join(base, "adapters"))
        base = os.path.dirname(base)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                adapters.setdefault(fn, p)
    return adapters
