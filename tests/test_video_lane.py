"""Video lane: ffmpeg-sampled frames + audio track, silent-video fallback, frame cap.

Run: python tests/test_video_lane.py   (embeds a few frames — needs ffmpeg; skips if absent)
"""
import os
import shutil
import subprocess
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

if not shutil.which("ffmpeg"):
    print("SKIP test_video_lane: ffmpeg not installed (mac/windows CI lanes)")
    sys.exit(0)


def check(name, cond, detail=None):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


def gen_video(path, audio=True):
    """3-second 320x240 testsrc, optional 440 Hz sine track. mpeg4+aac: stock codecs."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-shortest", "-c:v", "mpeg4", "-qscale:v", "5", "-c:a", "aac"]
    else:
        cmd += ["-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
                "-c:v", "mpeg4", "-qscale:v", "5"]
    cmd += [path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def kinds(c, uri):
    return c.execute("SELECT kind, span_start, span_end, thumb FROM assets WHERE uri=?",
                     (uri,)).fetchall()


root = tempfile.mkdtemp(prefix="mediarag-video-")
try:
    c = store.connect(root)
    mp4 = os.path.join(root, "clip_with_audio.mp4")
    silent = os.path.join(root, "clip_silent.m4v")
    gen_video(mp4)
    gen_video(silent, audio=False)

    check("kind_of maps video", ingest.kind_of(mp4) == "video" and ingest.kind_of(silent) == "video",
          ingest.kind_of(mp4))

    # --- with audio: frames + audio + label
    import unittest.mock as um
    with um.patch.object(ingest, "VIDEO_FPS", 1.0):
        n, skip = ingest.index_file(c, mp4)
        rows = kinds(c, mp4)
        frames = [(s, e) for k, s, e, _ in rows if k == "image" and s >= 0]
        auds = [(s, e) for k, s, e, _ in rows if k == "audio"]
        labels = [(k, s, e, t) for k, s, e, t in rows if s < 0]
        thumbs = [t for k, s, e, t in rows if k == "image" and s >= 0 and t is not None]
        check("audio-video indexed rows", n > 2 and not skip, (n, skip))
        check("frames present with timestamps", len(frames) >= 2 and frames[0][0] == 0.0, frames[:4])
        check("spans strictly increase", all(b[0] > a[0] for a, b in zip(frames, frames[1:])), frames[:6])
        check("frame period == 1/fps", abs(frames[0][1] - frames[0][0] - 1.0) < 0.01, frames[0])
        check("audio track indexed", len(auds) >= 1 and abs(auds[0][1] - 3.0) < 0.1, auds)
        check("thumbnails embedded", thumbs and all(t.startswith(b"\x89PNG") for t in thumbs),
              len(thumbs))
        check("label row", len(labels) == 1 and labels[0][3] is None, labels)

    # --- incremental: second pass skips via fingerprint
    n2, _ = ingest.index_file(c, mp4)
    check("incremental skip", n2 == -1, n2)

    # --- silent m4v: frames only
    with um.patch.object(ingest, "VIDEO_FPS", 1.0):
        n3, skip3 = ingest.index_file(c, silent)
        rows3 = kinds(c, silent)
        check("silent indexed", n3 >= 2 and not skip3, (n3, skip3))
        check("silent has frames", len([x for x in rows3 if x[0] == "image" and x[1] >= 0]) >= 2,
              len(rows3))
        check("silent has no audio rows", not [r for r in rows3 if r[0] == "audio"], rows3)

    # --- frame cap cut + warning rides the successful index
    capmp4 = os.path.join(root, "capped.mp4")
    gen_video(capmp4)
    with um.patch.object(ingest, "VIDEO_FPS", 1.0), um.patch.object(ingest, "VIDEO_MAX_FRAMES", 1):
        n4, skip4 = ingest.index_file(c, capmp4)
        capframes = [x for x in kinds(c, capmp4) if x[0] == "image" and x[1] >= 0]
        check("frame cap honored", len(capframes) == 1, len(capframes))
        check("frame_cap warning on success", n4 > 0 and len(skip4) == 1
              and skip4[0]["code"] == "frame_cap", (n4, skip4))

    # --- index_dir keeps counters clean while carrying the warning
    res = ingest.index_dir(c, root, incremental=True)
    check("index_dir counts unchanged files", res["unchanged"] >= 3 and res["added"] == 0, res)

    # --- search over video rows (real store, live engine)
    from core.engine import embed_doc
    hits = store.search(c, embed_doc("test pattern frame"), k=3)
    check("search over video rows returns results", len(hits) >= 1, len(hits))

    # --- FTS identity is kind-scoped: a frame match never hydrates the audio row
    fts_img = store.keyword_search(c, "frame", k=10)
    check("keyword hits for frame text are all image-kind",
          fts_img and all(h["kind"] == "image" for h in fts_img),
          [(h["uri"].split("/")[-1], h["kind"]) for h in fts_img])
    fts_aud = store.keyword_search(c, "frame", k=10, kind="audio")
    check("kind=audio filter returns no hydrating frame hits", fts_aud == [], fts_aud)
    both = store.rrf_blend([
        {"uri": mp4, "kind": "image", "span_start": 0.0, "span_end": 1.0,
         "snippet": "frame", "score": 0.9, "thumbnail": None},
        {"uri": mp4, "kind": "audio", "span_start": 0.0, "span_end": 3.0,
         "snippet": "audio", "score": 0.8, "thumbnail": None}], [], k=5)
    check("RRF keeps kind-distinct rows at the same uri+span", len(both) == 2,
          [(r["uri"].split("/")[-1], r["kind"]) for r in both])

    # --- warnings aggregation: skipped_total counts warned files, skipped does not
    with um.patch.object(ingest, "VIDEO_MAX_FRAMES", 1):
        capres = ingest.index_dir(c, root, incremental=False)
    check("index_dir reports warnings separately",
          capres["warnings"] >= 1 and capres["skips_total"] == capres["skipped"] + capres["warnings"]
          and capres["processed"] >= 2, capres)

    # --- audio infra failure aborts the file (no partial corpus, no fingerprint)
    import unittest.mock as um2
    boom = os.path.join(root, "boom.mp4")
    gen_video(boom)
    with um2.patch.object(ingest, "_audio_rows", side_effect=RuntimeError("model gone")):
        n6, skip6 = ingest.index_file(c, boom)
        check("audio model failure propagates (fail loud, no rows)",
              n6 == 0 and skip6 and skip6[0]["code"] == "read_error", (n6, skip6))
        fp_row = c.execute("SELECT COUNT(*) FROM fingerprints WHERE uri=?", (boom,)).fetchone()[0]
        row_cnt = c.execute("SELECT COUNT(*) FROM assets WHERE uri=?", (boom,)).fetchone()[0]
        check("no fingerprint left behind after failure", fp_row == 0 and row_cnt == 0,
              (fp_row, row_cnt))
    # a genuinely absent track still degrades to frames-only
    with um2.patch.object(ingest, "_audio_rows",
                          side_effect=__import__("subprocess").CalledProcessError(1, "ffmpeg")):
        n7, skip7 = ingest.index_file(c, boom)
        auds7 = [1 for k, s, e, t in kinds(c, boom) if k == "audio"]
        check("ffmpeg no-stream error degrades to frames-only",
              n7 >= 2 and not auds7 and not skip7, (n7, skip7))

    print("ALL PASS")
finally:
    shutil.rmtree(root, ignore_errors=True)
