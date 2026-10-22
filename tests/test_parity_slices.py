"""Self-check for parity slices: FTS keyword lane, RRF blend, doc bridge, label rows.

Run: python3 tests/test_parity_slices.py  (uses scratch corpus + real EG2 engine)
"""
import os
import subprocess
import shutil
import sys
try:  # pre-bind venv copies before engine prepends ~/pylibs2 (fleet env only)
    import regex  # noqa: F401
    import PIL  # noqa: F401 — pylibs2/PIL is a broken shadow (no _imaging)
except ImportError:  # clean envs (CI) have no pylibs2 shadow to guard against
    pass

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import ingest, search as search_mod, store  # noqa: E402

TMP_ROOT = os.path.join(os.path.expanduser("~/.cache/mediarag"), "parity-check")


def cleanup():
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


def setup_corpus():
    os.makedirs(f"{TMP_ROOT}/docs", exist_ok=True)
    # (1) label-miss case: name is a producer label, content unrelated
    with open(f"{TMP_ROOT}/docs/video_hd_20261010_c6_12s.wav.txt", "w") as f:
        f.write("Completely unrelated filler content about the weather in spring.\n\n" * 6)
    # (2) a docx via pandoc round-trip (bridge target)
    md = f"{TMP_ROOT}/docs/source.md"
    with open(md, "w") as f:
        f.write("# Ledger notes\n\nThe quarterly reconciliation is due Friday.\n\nSecond paragraph for chunking.")
    subprocess.run(["pandoc", md, "-o", f"{TMP_ROOT}/docs/ledger.docx"], check=True)
    # (3) plain text w/ exact keyword
    with open(f"{TMP_ROOT}/docs/zebra.md", "w") as f:
        f.write("The zebra escaped through the northern gate at dawn.")


def main():
    cleanup()
    setup_corpus()
    c = store.connect(TMP_ROOT)
    try:
        report = ingest.index_dir(c, TMP_ROOT + "/docs")
        print("ingest:", report)
        for s in report.get("skips", []):
            print("SKIP:", s)
        assert report["added"] >= 3, f"expected text+label rows for each file: {report}"
    finally:
        c.close()

    # (A) hybrid: zebra content via keyword lane beats weakly-related semantic fill
    hits = search_mod.search(TMP_ROOT, "zebra", k=3)
    assert any("zebra" in h["uri"].lower() for h in hits), f"keyword lane missed zebra: {hits}"
    assert hits[0].get("fused") or len(hits) == 1

    # (B) label lookup: "C6" must resolve the c6-named file (was the dogfood miss)
    hits = search_mod.search(TMP_ROOT, "where is the C6 clip", k=3)
    assert any("c6" in h["uri"].lower() for h in hits), f"label row failed: {hits}"

    # (C) doc bridge: query the docx content
    hits = search_mod.search(TMP_ROOT, "quarterly reconciliation due", k=3)
    assert any("ledger.docx" in h["uri"] for h in hits), f"docx bridge failed: {hits}"

    # (D) malformed fts query doesn't kill the semantic lane
    hits = search_mod.search(TMP_ROOT, '"unclosed quote AND (', k=3)
    assert isinstance(hits, list)

    print("ALL PARITY CHECKS PASS")


if __name__ == "__main__":
    main()
    cleanup()