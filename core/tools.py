"""MediaRAG tool surface: the five MCP/plugin tools."""
from __future__ import annotations

import os
import subprocess
import sys

from . import ingest as ing
from . import store
from . import search as se
from .urlsafe import resolve_for_fetch, url_problem  # noqa: F401  (url_problem re-exported for callers)

from .config_root import default_root as _default_root, config_root as _shared_config_root  # noqa: E402

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config_root() -> str:
    """Root from the package dir's mediarag-config.yaml (shared seam)."""
    return _shared_config_root(_PKG_DIR)


DEFAULT_ROOT = _default_root(_PKG_DIR)


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
        pin, why = resolve_for_fetch(target)
        if why or not pin:
            return {"code": "blocked_url", "url": target, "detail": why or "no usable address"}
        r = subprocess.run(
            ["curl", "-fsS", "--proto", "=http,https", "--max-time", "120",
             "--max-filesize", "268435456", "-w", "%{http_code}\t%{redirect_url}\t%{content_type}",
             "--resolve", pin,
             "-o", out, target],
            capture_output=True, timeout=130,
        )
        if r.returncode != 0:
            return {"code": "fetch_error", "url": target, "detail": r.stderr.decode()[-200:]}
        status, _, rest = r.stdout.decode().strip().partition("\t")
        nxt, _, ctype = rest.partition("\t")
        if not (status.startswith("3") and nxt):
            break
        target = nxt
    else:
        return {"code": "fetch_error", "url": url, "detail": "too many redirects"}
    size = os.path.getsize(out)
    if size > 256 * 1024 * 1024:
        os.unlink(out)
        return {"code": "too_large", "url": url, "size": size}
    if out.endswith(".bin"):
        # Extensionless URL (/document, /download?id=…): the server told us what
        # it sent, so name it by content type or ingest cannot sniff the format.
        ext = _CONTENT_EXT.get(ctype.split(";")[0].strip().lower())
        if ext:
            renamed = out[:-4] + ext
            os.replace(out, renamed)
            out = renamed
    return {"code": "ok", "path": out, "size": size}


_CONTENT_EXT = {
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/html": ".html",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "text/csv": ".csv",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def _fn_from_url(url: str) -> str:
    import hashlib
    import re
    tail = os.path.basename(url.split("?")[0]) or "bin"
    tail = re.sub(r"[^A-Za-z0-9._-]", "_", tail)[:64] or "bin"
    if "." not in tail:
        tail += ".bin"
    return f"{hashlib.sha1(url.encode()).hexdigest()[:10]}-{tail}"


class SecretScopeError(RuntimeError):
    """The Hermes secret scope was present but failed. Never fall back to raw env."""


_SCHEME_SECRETS = {"raindrop": "RAINDROP_API_TOKEN", "notion": "NOTION_TOKEN"}


def _secret(name: str) -> str | None:
    """Hermes path uses secret_scope (fail closed). Standalone MCP has no agent
    package, so the client-injected env is the token."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return os.environ.get(name) or None
    try:
        return get_secret(name) or None
    except Exception as ex:
        # Silently returning None here ran the adapter unauthenticated and
        # surfaced a confusing 401 instead of the real scope failure.
        raise SecretScopeError(f"secret scope failed for {name}: {ex}") from ex


def _adapter_env(scheme: str) -> dict:
    env = os.environ.copy()
    name = _SCHEME_SECRETS.get(scheme)
    if name is None:
        return env
    tok = _secret(name)  # only the token this adapter actually needs
    if tok:
        env[name] = tok
    else:
        env.pop(name, None)
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
        env = _adapter_env(scheme)
    except SecretScopeError as ex:
        return {"code": "secret_error", "scheme": scheme, "detail": str(ex)[:300]}
    timed_out = False
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired as ex:
        # A long sync that hits the cap has already emitted complete items.
        # Ingest those instead of throwing them away with the exception.
        timed_out = True
        stdout_b = ex.stdout or b""
        stderr_b = ex.stderr or b""
    except OSError as ex:                      # unreadable/removed adapter file
        return {"code": "adapter_error", "scheme": scheme, "stderr": str(ex)[-300:]}
    else:
        if r.returncode != 0:
            return {"code": "adapter_error", "scheme": scheme, "stderr": r.stderr.decode()[-300:]}
        stdout_b, stderr_b = r.stdout, r.stderr
    from .engine import embed_doc
    import json as _json
    added = 0
    skipped = []
    c = store.connect(root)
    try:
        for line in stdout_b.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = _json.loads(line)
            except ValueError:
                continue
            if "text" in item:
                # Chunk the body, not just embed it whole: a single row capped at
                # 1024 tokens lost everything past the cap, and a title-only
                # snippet left nothing for the keyword lane or the result view.
                rows = []
                for start, end, chunk in ing.text_chunks(item["text"]):
                    rows.append({"uri": item["uri"], "kind": "text",
                                 "span_start": start, "span_end": end,
                                 "snippet": chunk, "vec": embed_doc(chunk), "thumb": None})
                if not rows:
                    continue
                store.upsert_uri(c, item["uri"], rows)
                added += len(rows)
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
    result = {"added": added, "skipped": skipped,
              "adapter_log": [ln.strip()[:200] for ln in stderr_b.decode().splitlines() if ln.strip()]}
    if timed_out:
        result.update({"code": "adapter_timeout", "partial": True,
                       "detail": "adapter hit the 1800s cap; complete items ingested"})
    return result


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
