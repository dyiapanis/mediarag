"""MediaRAG tool surface: the five MCP/plugin tools."""
from __future__ import annotations

import os
import subprocess
import sys

from . import ingest as ing
from . import store
from . import search as se
from .urlsafe import url_problem

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config_root() -> str:
    """Corpus root from mediarag-config.yaml in the plugin/package dir.

    Same precedence as the plugin __init__'s TOOL_ROOT (config > MEDIARAG_ROOT env >
    default) so the native lane (explicit root=) and every defaulting caller —
    the stdio table, direct core use — resolve identically.
    # ponytail: flat `root:` line parse — no yaml dep; switch if the config grows keys.
    """
    path = os.path.join(_PKG_DIR, "mediarag-config.yaml")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("root:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


DEFAULT_ROOT = _config_root() or os.environ.get("MEDIARAG_ROOT") or os.path.expanduser("~/.mediarag")


def _root(root: str | None) -> str:
    return root or DEFAULT_ROOT


def tool_search(query: str, k: int = 5, root: str | None = None,
                include_thumbnails: bool = True, kind: str | None = None,
                **_x) -> dict:
    return {"results": se.search(_root(root), query, k=k, kind=kind,
                                 include_thumbnails=include_thumbnails)}


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
        # index_file returns -1 for "unchanged" — an internal sentinel index_dir
        # consumes. A public tool must never report a negative add count.
        added, skipped = ing.index_file(c, p)
        return {"added": max(added, 0), "skipped": skipped}
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
    s["source_adapters"] = list_adapters(_adapter_root())
    return s


# --- add_source lanes -------------------------------------------------------

_UA = {"User-Agent": "MediaRAG/1.0"}


_MAX_REDIRECTS = 5


def fetch_url(url: str) -> dict:
    """Download (≤256MB, 120s) to the persistent cache; return {'code','path'}.

    Redirects are followed here, not by curl, so every hop passes url_problem().
    """
    import shutil
    if not shutil.which("curl"):
        raise RuntimeError("curl not installed")
    out = os.path.join(ing.cache_dir(), "_url", _fn_from_url(url))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    target = url
    for _hop in range(_MAX_REDIRECTS + 1):
        why = url_problem(target)
        if why:
            return {"code": "blocked_url", "url": target, "detail": why}
        r = subprocess.run(
            ["curl", "-fsS", "--proto", "=http,https", "--max-time", "120",
             "--max-filesize", "268435456", "-w", "%{http_code} %{redirect_url}",
             "-o", out, target],
            capture_output=True, timeout=130,
        )
        if r.returncode != 0:
            return {"code": "fetch_error", "url": target, "detail": r.stderr.decode()[-200:]}
        status, _, nxt = r.stdout.decode().strip().partition(" ")
        if not (status.startswith("3") and nxt):
            break
        target = nxt
    else:
        return {"code": "fetch_error", "url": url, "detail": "too many redirects"}
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


def _raindrop_token() -> str | None:
    """Hermes path uses secret_scope (fail closed). Standalone MCP has no agent
    package, so the client-injected env is the token."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return os.environ.get("RAINDROP_API_TOKEN") or None
    try:
        return get_secret("RAINDROP_API_TOKEN") or None
    except Exception:
        return None


def _adapter_env() -> dict:
    env = os.environ.copy()
    tok = _raindrop_token()
    if tok:
        env["RAINDROP_API_TOKEN"] = tok
    else:
        env.pop("RAINDROP_API_TOKEN", None)
    return env


def run_adapter_sink(source: str, root: str) -> dict:
    """Adapter scheme URI → runner JSON-lines contract (sources.d)."""
    scheme, rest = source.split("://", 1)
    adapters = list_adapters(_adapter_root())
    adapter = adapters.get(scheme)
    if not adapter:
        return {"code": "unknown_scheme", "scheme": scheme,
                "available": sorted(adapters)}
    # Adapters are Python. Run them through this interpreter on every platform:
    # wheels and Windows checkouts carry no exec bit, and the shebang is not
    # something we can rely on. `sys.executable adapter` needs neither.
    cmd = [sys.executable, adapter, "fetch", rest]
    # stdout JSON lines: {"uri","title","text"} text items | {"uri","path"} file items
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800, env=_adapter_env())
    except OSError as ex:                      # unreadable/removed adapter file
        return {"code": "adapter_error", "scheme": scheme, "stderr": str(ex)[-300:]}
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
            elif "path" in item:
                if not os.path.exists(item["path"]):
                    skipped.append({"uri": item["uri"], "code": "missing_path",
                                    "detail": item["path"][:200]})
                    continue
                n, skip = ing.index_file(c, item["path"])
                added += max(n, 0)
                for s in skip:
                    s["uri"] = item["uri"]
                    skipped.append(s)
    finally:
        c.close()
    # stderr is the adapter's progress channel, not a failure list: reporting it
    # as "skipped" turned every successful run into a fake skip entry.
    return {"added": added, "skipped": skipped,
            "adapter_log": [ln.strip()[:200] for ln in r.stderr.decode().splitlines() if ln.strip()]}


def _adapter_root() -> str:
    """User adapters load only from the configured corpus root (mediarag-config.yaml /
    MEDIARAG_ROOT / default), never from a per-call `root`, so a tool argument cannot
    choose which executables run."""
    return os.path.expanduser(DEFAULT_ROOT)


def _adapter_dirs(root: str) -> list[str]:
    """<root>/sources.d then the package's own adapters/ — and nothing above it.

    The old walk-up used range(3) from core/, which resolved $HOME/adapters:
    anything executable dropped there silently became an invokable scheme.
    """
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return ([os.path.join(root, "sources.d")] if root else []) + \
           [os.path.join(package, "adapters")]


def list_adapters(root: str) -> dict:
    """<root>/sources.d/<scheme> user adapters + the shipped ./adapters/ ones."""
    adapters: dict[str, str] = {}
    for d in _adapter_dirs(root):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            # No exec-bit test: wheels and Windows checkouts don't carry one. Every
            # adapter runs through sys.executable anyway.
            if os.path.isfile(p) and not fn.startswith((".", "_")) and fn != "__init__.py":
                adapters.setdefault(fn, p)
    return adapters
