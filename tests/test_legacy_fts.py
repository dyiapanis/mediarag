"""A legacy kw_fts shape must be rebuilt, not kept by IF NOT EXISTS.

Regression for old corpora whose FTS table lacks span_start: every insert and
every keyword search failed closed against them.

Run: python tests/test_legacy_fts.py   (no model, no network)
"""
import os
import shutil
import sqlite3
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core import store  # noqa: E402


def check(name, cond, detail=None):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


root = tempfile.mkdtemp(prefix="mediarag-legacy-")
try:
    os.makedirs(os.path.join(root, ".corpus"))
    db = os.path.join(root, ".corpus", "store.db")
    raw = sqlite3.connect(db)
    raw.executescript("""
    CREATE TABLE assets(
        uri TEXT NOT NULL, kind TEXT NOT NULL,
        span_start REAL NOT NULL, span_end REAL NOT NULL,
        snippet TEXT NOT NULL, thumb BLOB, vec BLOB NOT NULL
    );
    CREATE UNIQUE INDEX assets_uri_span ON assets(uri, span_start);
    CREATE VIRTUAL TABLE kw_fts USING fts5(uri UNINDEXED, text);
    CREATE TABLE fingerprints(uri TEXT PRIMARY KEY, fp TEXT);
    """)
    uri = "/corpus/legacy.md"
    raw.execute("INSERT INTO assets VALUES(?,?,?,?,?,?,?)",
                (uri, "text", 0.0, 1.0, "marmalade lanterns in the legacy annex", None, b"\x00" * 16))
    raw.execute("INSERT INTO kw_fts VALUES(?,?)", (uri, "marmalade lanterns in the legacy annex"))
    raw.commit()
    raw.close()

    c = store.connect(root)
    cols = [r[1] for r in c.execute("PRAGMA table_info(kw_fts)")]
    check("legacy kw_fts rebuilt with span_start", cols == ["uri", "span_start", "text"], cols)
    hits = store.keyword_search(c, "marmalade", k=5)
    check("keyword search works on a migrated corpus", len(hits) == 1, hits)
    check("migrated hit is the legacy row", hits and hits[0]["uri"] == uri, hits)

    # inserts that used to fail closed now succeed
    rows = [{"uri": uri, "kind": "text", "span_start": 0.0, "span_end": 1.0,
             "snippet": "fresh lantern census after migration", "thumb": None,
             "vec": [0.0] * 8}]
    store.upsert_uri(c, uri, rows, fp="1:1")
    hits = store.keyword_search(c, "fresh", k=5)
    check("upsert works after migration", len(hits) == 1, hits)

    # re-opening must not rebuild or wipe a healthy table
    before = c.execute("SELECT COUNT(*) FROM kw_fts").fetchone()[0]
    c.close()
    c = store.connect(root)
    after = c.execute("SELECT COUNT(*) FROM kw_fts").fetchone()[0]
    check("healthy schema is left alone on reopen", before == after and after > 0, (before, after))
    c.close()
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\nALL LEGACY FTS CHECKS PASS")
