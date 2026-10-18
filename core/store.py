"""MediaRAG corpus store: SQLite + embeddings (f32 blobs), brute-force cosine.

Schema (the migration unit — unchanged from the validated corpus):
assets(uri, kind, span_start, span_end, snippet, thumb, vec) UNIQUE(uri, span_start).
"""
from __future__ import annotations

import base64
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
    """agent column (domain stamps, rag parity). Idempotent."""
    try:
        c.execute("ALTER TABLE assets ADD COLUMN agent TEXT NOT NULL DEFAULT ''")
        c.commit()
    except sqlite3.OperationalError:
        pass  # column exists
    c.execute("CREATE INDEX IF NOT EXISTS assets_agent ON assets(agent)")


def db_path(root: str) -> str:
    return os.path.join(root, ".corpus", "store.db")


def connect(root: str) -> sqlite3.Connection:
    os.makedirs(os.path.join(root, ".corpus"), exist_ok=True)
    c = sqlite3.connect(db_path(root))
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
            "INSERT OR REPLACE INTO assets(uri,kind,span_start,span_end,snippet,thumb,vec,agent) VALUES(?,?,?,?,?,?,?,?)",
            [
                (r["uri"], r["kind"], r["span_start"], r["span_end"], r["snippet"][:2000],
                 r.get("thumb"), enc(r["vec"]), r.get("agent", "")) for r in rows
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
                   kind: str | None = None, agent: str | None = None) -> list[dict]:
    """FTS5 BM25 hits, same result shape as search(). Empty query → []."""
    if not query.strip():
        return []
    sub = "SELECT DISTINCT uri FROM assets"
    clauses, subparams = [], []
    if kind:
        clauses.append("kind = ?")
        subparams.append(kind)
    if agent:
        clauses.append("agent = ?")
        subparams.append(agent)
    if clauses:
        sub += " WHERE " + " AND ".join(clauses)
    fts_q = _sanitize_fts_query(query)
    if not fts_q:
        return []
    sql = (f"SELECT uri, span_start FROM kw_fts WHERE kw_fts MATCH ? AND uri IN ({sub}) "
           "ORDER BY bm25(kw_fts) LIMIT ?")
    try:
        hits = c.execute(sql, (fts_q, ) + tuple(subparams) + (k, )).fetchall()
    except sqlite3.OperationalError:
        return []  # malformed query string (fts syntax chars) — semantic lane still answers
    out = []
    for uri, span in hits:
        r = c.execute(
            "SELECT kind, span_end, snippet, thumb FROM assets WHERE uri=? AND span_start=?",
            (uri, span)).fetchone()
        if not r:
            continue
        out.append({"uri": uri, "kind": r[0], "score": 0.0, "span_start": span,
                    "span_end": r[1], "snippet": r[2],
                    "thumbnail": ("data:image/png;base64," + base64.b64encode(r[3]).decode())
                    if r[3] else None})
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
           include_thumbnails: bool = True, agent: str | None = None) -> list[dict]:
    where = []
    params: list = []
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if agent:
        where.append("agent = ?")
        params.append(agent)
    sql = "SELECT uri,kind,span_start,span_end,snippet,vec,thumb FROM assets" \
        + (" WHERE " + " AND ".join(where) if where else "")
    rows = c.execute(sql, tuple(params)).fetchall()
    if not rows:
        return []
    mats = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
    dots = mats @ qvec
    top = sorted(range(len(rows)), key=lambda i: -dots[i])[:k]
    out = []
    for i in top:
        r = rows[i]
        thumb = None
        if include_thumbnails and r[6]:
            thumb = "data:image/png;base64," + base64.b64encode(r[6]).decode()
        out.append({
            "uri": r[0], "kind": r[1], "score": float(dots[i]),
            "span_start": r[2], "span_end": r[3], "snippet": r[4], "thumbnail": thumb,
        })
    return out


def stats(c: sqlite3.Connection) -> dict:
    dist = c.execute("SELECT kind, COUNT(*) FROM assets GROUP BY kind").fetchall()
    return {"total": sum(n for _, n in dist), "counts": {k: n for k, n in dist},
            "uris": c.execute("SELECT COUNT(DISTINCT uri) FROM assets").fetchone()[0]}