#!/usr/bin/env python3
"""End-to-end smoke: ingest -> embed -> search -> thumbnail, on a scratch corpus.

Direct-run: python3 tests/test_smoke_e2e.py  (exit 0 = all pass)

This is the one test that proves the platform can actually *run* the engine, as
opposed to importing it. Every CI lane runs it; md + png only, so it needs no
ffmpeg/pandoc/pdftotext and no network beyond the pinned model cache.
"""
import base64
import os
import shutil
import sys
import tempfile

try:  # pre-bind venv copies before engine prepends ~/pylibs2 (fleet env only)
    import regex  # noqa: F401
    import PIL  # noqa: F401 — pylibs2/PIL is a broken shadow (no _imaging)
except ImportError:  # clean envs (CI) have no pylibs2 shadow to guard against
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PIL import Image  # noqa: E402

from core import tools  # noqa: E402

PASSED = FAILED = 0
DOC_LINE = "The quokka census flagged seventeen marmalade lanterns in the annex."
NOTE_LINE = "Quokka lantern budget approved by the annex committee on Thursday."


def check(name, ok, detail=None):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}  {detail}")


def thumb_bytes(value):
    """Result thumbnails are `data:image/png;base64,…` URLs; tolerate raw bytes."""
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    payload = value.split(",", 1)[-1]
    return base64.b64decode(payload + "=" * (-len(payload) % 4))


def main():
    root = tempfile.mkdtemp(prefix="mediarag-smoke-")
    try:
        corpus = os.path.join(root, "docs")
        os.makedirs(corpus)

        with open(os.path.join(corpus, "census.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# Annex notes\n\n{DOC_LINE}\n\nFiller so the chunk clears the empty-file floor.\n")
        # A flat colour is the easiest thing for the vision tower to match, and it
        # proves Pillow can open, resize and re-encode on this platform.
        Image.new("RGB", (320, 200), (222, 92, 40)).save(os.path.join(corpus, "swatch.png"))

        note = tools.tool_add_note(NOTE_LINE, root=root)
        check("add_note returns a note uri", str(note.get("uri", "")).startswith("note://"), note)

        ingested = tools.tool_ingest_dir(corpus, root=root)
        check("ingest adds both files", ingested.get("added", 0) >= 2 and not ingested.get("skipped"), ingested)

        status = tools.tool_status(root=root)
        counts = status.get("counts") or {}
        check("status counts match the corpus",
              status.get("uris") == 3 and counts.get("text", 0) >= 1 and counts.get("image", 0) >= 1, status)

        hits = tools.tool_search("marmalade lanterns annex", k=3, root=root)["results"]
        check("document is top-1 by content",
              hits and hits[0]["uri"].endswith("census.md"), [h["uri"] for h in hits[:2]])
        check("document snippet carries matched text",
              hits and "lantern" in hits[0]["snippet"].lower(), hits and hits[0]["snippet"][:80])

        note_hits = tools.tool_search("lantern budget committee", k=3, root=root)["results"]
        check("note is retrievable", note_hits and note_hits[0]["uri"].startswith("note://"),
              [h["uri"] for h in note_hits[:2]])

        image_hits = tools.tool_search("orange colour swatch", k=3, root=root, kind="image")["results"]
        thumbs = [thumb_bytes(h["thumbnail"]) for h in image_hits]
        check("image hit is returned", bool(image_hits), image_hits)
        check("image hit carries a real PNG thumbnail",
              any(t.startswith(b"\x89PNG\r\n\x1a\n") for t in thumbs),
              [len(t) for t in thumbs])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
