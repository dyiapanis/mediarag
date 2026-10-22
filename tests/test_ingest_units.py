#!/usr/bin/env python3
"""P2 unit tests: chunk ports, store integrity, adapter discovery. No model needed.

Direct-run: python3 tests/test_ingest_units.py (exit 0 = all pass).
Adapted from the independent reviewer's probes (review_mediarag/probe.py).
"""
import json
import os
import shutil
import sys
try:  # pre-bind venv copies before engine prepends ~/pylibs2 (fleet env only)
    import regex  # noqa: F401
    import PIL  # noqa: F401 — pylibs2/PIL is a broken shadow (no _imaging)
except ImportError:  # clean envs (CI) have no pylibs2 shadow to guard against
    pass
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ingest, store, tools  # noqa: E402  (P3 package layout: core/, adapters/, scripts/)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSED = FAILED = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}  {detail}")


def main() -> int:
    # OOM regression (2026-10-19 fleet incident): a no-paragraph-break file
    # (81KB calibre csv) produced ONE giant chunk → unbounded token sequence →
    # 12GB attention blowup. Both layers must hold: chunker splits oversized
    # paragraphs; engine caps tokens at MAX_TOK.
    from core import engine  # numpy-only usage; no model load needed for the cap
    giant = "word " * 20000  # 100KB single paragraph, no \n\n
    chunks = ingest.text_chunks(giant)
    check("oversized paragraph splits into many chunks", len(chunks) > 10, f"{len(chunks)} chunks")
    check("every chunk within chunk budget", all(len(c[2]) <= ingest.CHUNK_CHARS + 1200 for c in chunks),
          f"max len {max(len(c[2]) for c in chunks)}")
    joined = " ".join(c[2] for c in chunks)
    check("chunks reconstruct the text losslessly", joined == giant.strip(), f"{len(joined)} vs {len(giant.strip())}")
    feeds = {"input_ids": np.arange(50000).reshape(1, -1), "attention_mask": np.ones((1, 50000))}
    capped = engine._cap_tokens(feeds)
    check("token cap slices to MAX_TOK", capped["input_ids"].shape == (1, engine.MAX_TOK)
          and capped["attention_mask"].shape == (1, engine.MAX_TOK), str(capped["input_ids"].shape))
    small = engine._cap_tokens({"input_ids": np.zeros((1, 5))})
    check("token cap leaves short inputs alone", small["input_ids"].shape == (1, 5), str(small["input_ids"].shape))

    tmp = tempfile.mkdtemp(prefix="mrtest-")
    try:
        # --- D1: text chunk spans are real char offsets, distinct per block ---
        t = ingest.text_chunks("x" * 700 + "\n\n" + "y" * 700 + "\n\n" + "z" * 100)
        check("text_chunks groups 3 paras to 2 blocks", len(t) == 2, str([(s, e) for s, e, _ in t]))
        spans = [(s, e) for s, e, _ in t]
        check("text_chunks offsets distinct (D1 fix)", len({s for s, _ in spans}) == 2, str(spans))
        check("text_chunks offsets ordered", spans[0][1] < spans[1][0], str(spans))
        # offsets slice back to the original text
        doc = "a" * 700 + "\n\n" + "b" * 700 + "\n\n" + "c" * 100
        for s, e, text in ingest.text_chunks(doc):
            check("slice(s,e) starts with chunk text", doc[int(s):int(e)].lstrip().startswith(text[:40].lstrip()),
                  f"[{s},{e}] {text[:40]!r}")

        # --- D3: subtitle timestamp fidelity ---
        check("_parse_ts SRT ms", abs(ingest._parse_ts("00:01:23,400") - 83.4) < 1e-6,
              str(ingest._parse_ts("00:01:23,400")))
        check("_parse_ts VTT ms", abs(ingest._parse_ts("00:01:23.400") - 83.4) < 1e-6,
              str(ingest._parse_ts("00:01:23.400")))
        check("_parse_ts mm:ss.m form", abs(ingest._parse_ts("01:23.4") - 83.4) < 1e-6,
              str(ingest._parse_ts("01:23.4")))
        srt = "1\n00:00:01,500 --> 00:00:03,200\nhello\n\n2\n00:00:04,000 --> 00:00:05,500\nsecond cue line\n"
        sc = ingest.subs_chunks(srt)
        check("subs 1 block, ms-accurate span", len(sc) == 1
              and abs(sc[0][0] - 1.5) < 1e-6 and abs(sc[0][1] - 5.5) < 1e-6, str(sc[:1]))

        # --- D4: wav sampwidth guard ---
        p24 = os.path.join(tmp, "pcm24.wav")
        with wave.open(p24, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(3)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00\x00" * 16000)  # 1s @16k 24-bit (3 bytes/sample)
        d = ingest.wav16_mono(p24)
        check("24-bit wav length == 1s, not garbage 1.5x", len(d) == 16000, f"len={len(d)}")

        # --- D2: shipped adapters discovered from repo layout ---
        ad = tools.list_adapters(tmp)
        check("adapters discovered", {"raindrop", "pcloud", "obsidian", "rclone"} <= set(ad), str(ad))
        check("adapter paths exist and are executable", all(os.access(v, os.X_OK) for v in ad.values()),
              str(ad))

        # --- store: D1 store-level regression (multi-chunk uri survives) ---
        c = store.connect(tmp)
        rows = [
            {"uri": "/f.md", "kind": "text", "span_start": 0.0, "span_end": 800.0,
             "snippet": "one", "vec": np.ones(4, dtype=np.float32), "thumb": None},
            {"uri": "/f.md", "kind": "text", "span_start": 1400.0, "span_end": 2000.0,
             "snippet": "two", "vec": -np.ones(4, dtype=np.float32), "thumb": None},
        ]
        added = store.upsert_uri(c, "/f.md", rows)
        n = c.execute("SELECT COUNT(*) FROM assets WHERE uri = ?", ("/f.md",)).fetchone()[0]
        check("multi-chunk uri keeps every block", added == 2 and n == 2, f"added={added} n={n}")
        # re-upsert replaces cleanly
        store.upsert_uri(c, "/f.md", [rows[0]])
        n2 = c.execute("SELECT COUNT(*) FROM assets WHERE uri = ?", ("/f.md",)).fetchone()[0]
        check("re-upsert replaces all rows for uri", n2 == 1, f"n={n2}")
        # upsert failure rolls back the delete
        class Boom(dict):
            pass
        bad = [rows[0], {"uri": "/f.md", "kind": "text", "span_start": 999.0, "span_end": 0.0,
              "snippet": "short", "vec": "not-bytes", "thumb": None}]
        try:
            store.upsert_uri(c, "/f.md", bad)
            rolled = False
        except Exception:
            rolled = True
        n3 = c.execute("SELECT COUNT(*) FROM assets WHERE uri = ?", ("/f.md",)).fetchone()[0]
        check("failed upsert rolls back delete (old rows intact)", rolled and n3 == 1, f"n={n3} rolled={rolled}")
        # knn search returns both-chunk dataset properly ordered
        q = np.ones(4, dtype=np.float32)
        hits = store.search(c, q, k=2)
        check("knn orders matching-first", hits and hits[0]["snippet"] == "one", str(hits[:1]))
        st = store.stats(c)
        check("stats shape", st["total"] == 1 and st["counts"].get("text") == 1, str(st))
        c.close()

        # --- audio chunk boundaries (pure function, no model): tiny 0.5s wav ---
        pw = os.path.join(tmp, "t.wav")
        with wave.open(pw, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.zeros(8000, dtype=np.int16) + 100).tobytes())
        data = ingest.wav16_mono(pw)
        blocks = [(i / ingest.SR, min((i + ingest.AUDIO_CHUNK) / ingest.SR, len(data) / ingest.SR))
                  for i in range(0, len(data), ingest.AUDIO_CHUNK)]
        check("tiny wav = one chunk [0,0.5]", len(blocks) == 1 and abs(blocks[0][1] - 0.5) < 1e-6, str(blocks))

        # --- skip codes from index_file (unsupported/empty paths without model) ---
        c = store.connect(tmp)
        p_bad = os.path.join(tmp, "weird.xyz")
        open(p_bad, "w").write("hello")
        n, skip = ingest.index_file(c, p_bad)
        check("unsupported ext skipped with code", n == 0 and skip[0]["code"] == "unsupported_ext", str(skip))
        p_empty = os.path.join(tmp, "empty.md")
        open(p_empty, "w").write("\n\n")
        n, skip = ingest.index_file(c, p_empty)
        check("empty file skipped with code empty", n == 0 and skip[0]["code"] == "empty", str(skip))
        c.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())