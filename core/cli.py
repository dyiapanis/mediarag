"""MediaRAG CLI: init / index / search / doctor (+ serve for the MCP client lane)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mediarag import tools as t  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="mediarag")
    ap.add_argument("--root", default=t.DEFAULT_ROOT, help="corpus root (default ~/.mediarag)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create an empty corpus at --root")
    p_index = sub.add_parser("index", help="bulk-index a directory")
    p_index.add_argument("path")
    sub.add_parser("doctor", help="health check (corpus, model, ffmpeg, adapters)")
    p_search = sub.add_parser("search", help="natural-language search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--json", action="store_true")
    sub.add_parser("status")

    a = ap.parse_args()
    root = a.root
    if a.cmd == "init":
        from . import store  # core.store
        store.connect(root)
        print("initialized", root)
        return 0
    if a.cmd == "index":
        r = t.tool_ingest_dir(a.path, root)
        print(json.dumps(r))
        return 0
    if a.cmd == "search":
        r = t.tool_search(a.query, k=a.k, root=root, include_thumbnails=False)
        if a.json:
            print(json.dumps(r, indent=2))
            return 0
        for hit in r["results"]:
            span = f" @ {hit['span_start']:.1f}-{hit['span_end']:.1f}s" if hit["span_end"] else ""
            print(f"{hit['score']:.3f}  {hit['uri']}{span}\n      {hit['snippet'][:140]}")
        return 0
    if a.cmd == "status":
        print(json.dumps(t.tool_status(root), indent=2))
        return 0
    if a.cmd == "doctor":
        checks = {}
        try:
            from . import store  # core.store
            c = store.connect(root)
            checks["corpus"] = store.stats(c)
            c.close()
        except Exception as e:
            checks["corpus"] = {"error": str(e)}
        try:
            checks["model"] = _model_check()
        except Exception as e:
            checks["model"] = {"error": str(e)}
        import shutil
        checks["ffmpeg"] = bool(shutil.which("ffmpeg"))
        checks["adapters"] = t.list_adapters(root)
        print(json.dumps(checks, indent=2))
        return 0
    return 1


def _model_check() -> dict:
    from mediarag.engine import embed_query, snap_dir
    import numpy as np
    v = embed_query("doctor probe")
    n = float(np.linalg.norm(v))
    assert n > 0.99 and n < 1.01, "query vector not normalized"
    return {"snapshot": snap_dir(), "dims": int(v.shape[0]), "norm": round(n, 4)}


if __name__ == "__main__":
    raise SystemExit(main())