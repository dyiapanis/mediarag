"""MediaRAG search: query embed + store KNN."""
from __future__ import annotations

import numpy as np

from . import store
from .engine import embed_query


def search(root: str, query: str, k: int = 5, kind: str | None = None,
           include_thumbnails: bool = True) -> list[dict]:
    """Hybrid: semantic (EG2 cosine) + keyword (FTS5 BM25) fused by RRF.

    """
    qvec = embed_query(query)
    c = store.connect(root)
    try:
        semantic = store.search(c, qvec, k=max(k, 10), kind=kind,
                                include_thumbnails=include_thumbnails)
        keyword = store.keyword_search(c, query, k=max(k, 10), kind=kind)
        blended = store.rrf_blend(semantic, keyword, k=k)
        # semantic-only corpora (no fts rows, e.g. legacy stores) fall back cleanly
        return blended if blended else semantic
    finally:
        c.close()


def add_note(root: str, text: str) -> dict:
    from .engine import embed_doc

    uri = f"note://local/{_slug(text)}"
    v = embed_doc(text)
    c = store.connect(root)
    try:
        row = {"uri": uri, "kind": "text", "span_start": 0.0, "span_end": 0.0,
               "snippet": text, "vec": v, "thumb": None}
        store.upsert_uri(c, uri, [row])
        return {"uri": uri, "rows": 1}
    finally:
        c.close()


def _slug(text: str) -> str:
    import hashlib
    import datetime
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{h}"