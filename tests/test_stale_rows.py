"""Re-ingest must not leave stale rows: fingerprints, deleted files, deleted sources.

Run: python tests/test_stale_rows.py   (embeds a few short texts, no other tools)
"""
import os
import shutil
import sys
import tempfile

try:  # pre-bind venv copies before engine prepends ~/pylibs2 (fleet env only)
    import regex  # noqa: F401
    import PIL  # noqa: F401
except ImportError:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core import ingest, store  # noqa: E402

TEXT = ("The quokka census flagged seventeen marmalade lanterns in the annex. "
        "Every lantern was counted twice by the Thursday committee. ") * 4


def check(name, cond, detail=None):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


def counts(c, uri):
    a = c.execute("SELECT COUNT(*) FROM assets WHERE uri=?", (uri,)).fetchone()[0]
    f = c.execute("SELECT COUNT(*) FROM fingerprints WHERE uri=?", (uri,)).fetchone()[0]
    k = c.execute("SELECT COUNT(*) FROM kw_fts WHERE uri=?", (uri,)).fetchone()[0]
    return a, f, k


root = tempfile.mkdtemp(prefix="mediarag-stale-")
try:
    corpus = os.path.join(root, "docs")
    other = os.path.join(root, "elsewhere")
    os.makedirs(corpus)
    os.makedirs(other)
    c = store.connect(root)

    doc = os.path.join(corpus, "census.md")
    with open(doc, "w") as fh:
        fh.write(TEXT)
    res = ingest.index_dir(c, corpus)
    check("ingest adds rows", res["added"] > 0, res)
    check("fingerprint recorded", counts(c, doc)[1] == 1, counts(c, doc))

    # 1. a re-ingest that passes fp=None must not leave a skip-if-unchanged trap
    # Clear the rows the way a failed or partial re-ingest does, with no
    # fingerprint supplied.
    c.execute("DELETE FROM assets WHERE uri=?", (doc,))
    c.execute("DELETE FROM kw_fts WHERE uri=?", (doc,))
    store.upsert_uri(c, doc, [], fp=None)
    check("fp=None clears the fingerprint", counts(c, doc)[1] == 0, counts(c, doc))
    n, _skip = ingest.index_file(c, doc)
    check("next ingest is not skipped as unchanged", n > 0, n)
    check("rows restored", counts(c, doc)[0] > 0, counts(c, doc))

    # 2. deleting a file reaps its rows on the next ingest of that directory
    remote = "pcloud://library/notes.md"
    c.execute("INSERT INTO assets(uri,kind,span_start,span_end,snippet,thumb,vec) "
              "VALUES(?,?,?,?,?,?,?)", (remote, "text", -1.0, -1.0, "remote note", None, b"\x00" * 16))
    c.execute("INSERT INTO kw_fts(uri,span_start,text) VALUES(?,?,?)",
              (remote, -1.0, "remote note"))
    c.commit()
    os.remove(doc)
    res = ingest.index_dir(c, corpus)
    check("deleted file is reported as reaped", res["reaped"] == 1, res)
    check("deleted file rows are gone", counts(c, doc) == (0, 0, 0), counts(c, doc))
    check("remote URIs are never reaped", counts(c, remote)[0] == 1, counts(c, remote))

    # 3. ingesting a subdirectory does not reap documents outside it
    outside = os.path.join(other, "keep.md")
    with open(outside, "w") as fh:
        fh.write(TEXT)
    ingest.index_dir(c, other)
    check("second source indexed", counts(c, outside)[0] > 0, counts(c, outside))
    res = ingest.index_dir(c, corpus)
    check("outside documents survive a partial ingest", counts(c, outside)[0] > 0,
          counts(c, outside))

    # 4. deleting a whole source directory removes it from the index
    shutil.rmtree(other)
    res = ingest.index_dir(c, root)
    check("deleted source is reaped", counts(c, outside) == (0, 0, 0), counts(c, outside))
    c.close()
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\nALL STALE ROW CHECKS PASS")
