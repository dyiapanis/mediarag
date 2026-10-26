"""MediaRAG corpus store: SQLite + embeddings (f32 blobs), brute-force cosine.

Schema (the migration unit — unchanged from the validated corpus):
assets(uri, kind, span_start, span_end, snippet, thumb, vec) UNIQUE(uri, span_start).
"""
from __future__ import annotations

import base64
import heapq
import os
import re
import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets(
    uri TEXT NOT NULL,
    kind TEXT NOT NULL,
    span_start REAL NOT NULL,
    span_end REAL NOT NULL,
    snippet TEXT NOT NULL,
    thumb BLOB,
    vec BLOB NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS assets_uri_span ON assets(uri, span_start);
CREATE INDEX IF NOT EXISTS assets_kind ON assets(kind);
CREATE VIRTUAL TABLE IF NOT EXISTS kw_fts USING fts5(
    uri UNINDEXED, span_start UNINDEXED, text
);
CREATE TABLE IF NOT EXISTS fingerprints(
    uri TEXT PRIMARY KEY, fp TEXT
);
"""


def _migrate(c: sqlite3.Connection) -> None:
    """Drop the legacy `agent` lane-stamp column if present (removed 2026-10-19:
    semantic search does the heavy lifting — no path-prefix tagging). Idempotent."""
    cols = [r[1] for r in c.execute("PRAGMA table_info(assets)")]
    if "agent" in cols:
        c.execute("DROP INDEX IF EXISTS assets_agent")  # index must go before the column
        c.execute("ALTER TABLE assets DROP COLUMN agent")
        c.commit()


def db_path(root: str) -> str:
    return os.path.join(root, ".corpus", "store.db")


def connect(root: str) -> sqlite3.Connection:
    os.makedirs(os.path.join(root, ".corpus"), exist_ok=True)
    c = sqlite3.connect(db_path(root), timeout=5.0)
    # WAL: the nightly indexer writes file-by-file while the gateway serves reads.
    # Rollback-journal mode made each side block the other for the whole build.
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    c.executescript(SCHEMA)
    _migrate(c)
    return c


def enc(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def dec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert_uri(c: sqlite3.Connection, uri: str, rows: list[dict], fp: str | None = None) -> int:
    """Replace every row for uri, then insert the new ones. Returns rows added."""
    c.execute("DELETE FROM assets WHERE uri = ?", (uri,))
    c.execute("DELETE FROM kw_fts WHERE uri = ?", (uri,))  # FTS5 mirror, stays in lockstep
    if fp is not None:
        c.execute("DELETE FROM fingerprints WHERE uri = ?", (uri,))
    try:
        c.executemany(
            "INSERT OR REPLACE INTO assets(uri,kind,span_start,span_end,snippet,thumb,vec) VALUES(?,?,?,?,?,?,?)",
            [
                (r["uri"], r["kind"], r["span_start"], r["span_end"], r["snippet"][:2000],
                 r.get("thumb"), enc(r["vec"])) for r in rows
            ],
        )
        c.executemany(
            "INSERT INTO kw_fts(uri, span_start, text) VALUES(?,?,?)",
            [(uri, r["span_start"], _fts_text(r["snippet"], r["uri"],
                                              is_label=r["span_start"] < 0)) for r in rows],
        )
        if fp:
            c.execute("INSERT INTO fingerprints(uri, fp) VALUES(?,?)", (uri, fp))
        c.commit()
    except Exception:
        c.rollback()
        raise
    return len(rows)


def _fts_text(snippet: str, uri: str, is_label: bool = False) -> str:
    """FTS text: label rows (span -1) carry filename+dir for producer-label
    lookups ('C6'); content rows carry snippet only — otherwise every audio
    chunk of a file matches its own filename and crowds the top-k."""
    if is_label:
        return f"{os.path.basename(uri)} {os.path.basename(os.path.dirname(uri))} {snippet}"
    return snippet


def _sanitize_fts_query(query: str) -> str:
    """Quote-wrap each word as an OR'd FTS5 prefix term.

    Quoting neutralises operator syntax in user input (AND/OR/NOT/NEAR);
    OR'd terms preserve recall ("where is the C6 clip" no longer AND-deads).
    Ported from rag query.py — same semantics, same ponytail caveat: OR'd
    prefix terms can over-match; tighten to plain quoted terms if ranking
    noise becomes a complaint.
    """
    terms = [t for t in re.split(r"\W+", query, flags=re.UNICODE) if t]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"*' for t in terms)


def keyword_search(c: sqlite3.Connection, query: str, k: int = 5,
                   kind: str | None = None, include_thumbnails: bool = True) -> list[dict]:
    """FTS5 BM25 hits, same result shape as search(). Empty query → []."""
    if not query.strip():
        return []
    fts_q = _sanitize_fts_query(query)
    if not fts_q:
        return []
    # One JOIN replaces a DISTINCT-uri subquery over all of assets plus a
    # per-hit lookup (10 extra queries at k=10 on a 770MB db).
    thumb_col = ", a.thumb" if include_thumbnails else ""
    sql = ("SELECT a.uri, a.span_start, a.kind, a.span_end, a.snippet" + thumb_col +
           " FROM kw_fts JOIN assets a"
           " ON a.uri = kw_fts.uri AND a.span_start = kw_fts.span_start"
           " WHERE kw_fts MATCH ?" + (" AND a.kind = ?" if kind else "") +
           " ORDER BY bm25(kw_fts) LIMIT ?")
    args: list = [fts_q] + ([kind] if kind else []) + [k]
    try:
        hits = c.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []  # malformed query string (fts syntax chars) — semantic lane still answers
    out = []
    for r in hits:
        thumb = None
        if include_thumbnails and r[5]:
            thumb = "data:image/png;base64," + base64.b64encode(r[5]).decode()
        out.append({"uri": r[0], "kind": r[2], "score": 0.0, "span_start": r[1],
                    "span_end": r[3], "snippet": r[4], "thumbnail": thumb})
    return out


def rrf_blend(semantic: list[dict], keyword: list[dict], k: int = 5,
              k60: int = 60) -> list[dict]:
    """Reciprocal-rank fusion of both lanes → top-k by fused score."""
    scores: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    for rank, hit in enumerate(semantic):
        key = (hit["uri"], hit["span_start"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k60 + rank + 1)
        rows.setdefault(key, hit)
    for rank, hit in enumerate(keyword):
        key = (hit["uri"], hit["span_start"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k60 + rank + 1)
        rows.setdefault(key, hit)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for key, s in top:
        h = dict(rows[key])
        h["score"] = float(s)
        h["fused"] = True
        out.append(h)
    return out


def search(c: sqlite3.Connection, qvec: np.ndarray, k: int = 5, kind: str | None = None,
           include_thumbnails: bool = True) -> list[dict]:
    """Top-k cosine hits. Scans vectors as a stream, hydrates only the winners.

    # ponytail: still a full brute-force scan (no ANN index) — ~0.4s at 71k rows.
    # Add HNSW/FAISS when the corpus passes ~500k rows.
    """
    where = []
    params: list = []
    if kind:
        where.append("kind = ?")
        params.append(kind)
    sql = "SELECT rowid, vec FROM assets" \
        + (" WHERE " + " AND ".join(where) if where else "")
    # heapq.nlargest evaluates the key once per row, so nothing but the top-k
    # is held: the old fetchall+np.stack built a 71k x 768 matrix (~440MB peak)
    # per query, against a 4GB cgroup.
    top = heapq.nlargest(
        k, c.execute(sql, tuple(params)),
        key=lambda r: float(np.frombuffer(r[1], dtype=np.float32) @ qvec),
    )
    thumb_col = ", thumb" if include_thumbnails else ""
    out = []
    for rowid, _vec in top:
        r = c.execute(
            "SELECT uri, kind, span_start, span_end, snippet" + thumb_col +
            " FROM assets WHERE rowid = ?", (rowid,)).fetchone()
        if r is None:
            continue
        thumb = None
        if include_thumbnails and r[5]:
            thumb = "data:image/png;base64," + base64.b64encode(r[5]).decode()
        out.append({
            "uri": r[0], "kind": r[1],
            "score": float(np.frombuffer(_vec, dtype=np.float32) @ qvec),
            "span_start": r[2], "span_end": r[3], "snippet": r[4], "thumbnail": thumb,
        })
    out.sort(key=lambda h: -h["score"])
    return out


def stats(c: sqlite3.Connection) -> dict:
    dist = c.execute("SELECT kind, COUNT(*) FROM assets GROUP BY kind").fetchall()
    return {"total": sum(n for _, n in dist), "counts": {k: n for k, n in dist},
            "uris": c.execute("SELECT COUNT(DISTINCT uri) FROM assets").fetchone()[0]}