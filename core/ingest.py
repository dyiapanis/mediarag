"""MediaRAG ingest: walk → parse → chunk → embed → store.

Ports the validated chunking: text = paragraphs grouped to ~1500-char blocks
(spans 0,0); srt/vtt = consecutive cues grouped to ~600-char blocks with real
[start,end] spans; images = whole file (+256px thumb); audio = 16k mono
(ffmpeg when not a 16k wav) chunked at 179,200 samples (11.2 s) with offsets.
"""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

from . import store
from .engine import embed_audio, embed_doc, embed_image

TEXT_EXT = {"md", "txt"}
SUBS_EXT = {"srt", "vtt"}
IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
AUDIO_EXT = {"wav", "mp3", "m4a", "ogg", "flac", "aac"}
# Document formats via external converters (parity with the fleet doc-RAG corpus):
#   pandoc: docx odt epub rst html csv rtf
#   pdftotext (poppler): pdf
DOC_PANDOC_EXT = {"docx", "odt", "epub", "rst", "html", "htm", "csv", "rtf"}
DOC_PDF_EXT = {"pdf"}

CHUNK_CHARS = 1500
SUBS_CHUNK_CHARS = 600
AUDIO_CHUNK = 179_200  # 11.2 s @ 16k
SR = 16_000
MIN_CONTENT_CHARS = 50  # rag parity: files whose extracted text is shorter are cruft

# rag-sources.yaml exclusions, mirrored verbatim (fleet corpus parity)
EXCLUDE_PATTERNS = (
    "*/.obsidian/*", "*/.trash/*", "*/.hub/*", "*/cache/*", "*/cache_documents/*",
    "*/agent-state-exports-flat/*", "*/.DS_Store", "*.tmp", "*.bak", "*.partial",
    "*.db", "*.db-wal", "*.db-shm", "*.enc", "*.eml",
)


def excluded(rel_path: str) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(rel_path, pat) for pat in EXCLUDE_PATTERNS)


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower().lstrip(".")


def kind_of(path: str) -> str | None:
    e = ext_of(path)
    if e in TEXT_EXT or e in SUBS_EXT or e in DOC_PANDOC_EXT or e in DOC_PDF_EXT:
        return "text"
    if e in IMAGE_EXT:
        return "image"
    if e in AUDIO_EXT:
        return "audio"
    return None


def doc_text(path: str) -> str:
    """Extract plain text from a document file via pandoc/pdftotext.

    Returns "" when the converter yields nothing (caller records `empty`).
    Raises RuntimeError with the tool name when a converter is missing (caller
    records the failure — never fatal to the walk).
    """
    import shutil
    e = ext_of(path)
    if e in DOC_PDF_EXT:
        tool = shutil.which("pdftotext")
        if not tool:
            raise RuntimeError("pdftotext not installed — needed for pdf ingestion")
        r = subprocess.run([tool, "-q", path, "-"], capture_output=True, text=True, timeout=120)
        return r.stdout
    if e in DOC_PANDOC_EXT:
        tool = shutil.which("pandoc")
        if not tool:
            raise RuntimeError("pandoc not installed — needed for document ingestion")
        r = subprocess.run([tool, path, "-t", "plain"], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pandoc failed: {r.stderr.strip()[:200]}")
        out = r.stdout
        if out.startswith("PK\x03\x04"):
            raise RuntimeError("pandoc emitted zip-binary (no reader for this format) — skipping")
        return out
    return ""


def text_chunks(s: str) -> list[tuple[float, float, str]]:
    """Paragraph blocks (~1500 chars) with real char offsets [start, end) in s."""
    out: list[tuple[float, float, str]] = []
    buf, buf_start, buf_end = "", None, 0
    pos = 0
    for p in s.split("\n\n"):
        start = s.find(p, pos)
        if start >= 0:
            pos = start  # paragraphs are non-overlapping in order
        t = p.strip()
        if not t:
            continue
        if buf and len(buf) + len(t) > CHUNK_CHARS:
            assert buf_start is not None
            out.append((float(buf_start), float(buf_end), buf))
            buf, buf_start = "", None
        if buf:
            buf += "\n\n" + t
        else:
            buf, buf_start = t, start
        buf_end = start + len(t)
    if buf:
        assert buf_start is not None
        out.append((float(buf_start), float(buf_end), buf))
    return out


def _parse_ts(t: str) -> float | None:
    """'00:01:23,400' | '00:01:23.400' | '01:23.4' → seconds."""
    t = t.replace(",", ".")
    m = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if len(m) >= 3:
        return m[0] * 3600 + m[1] * 60 + m[2]
    if len(m) == 2:
        return m[0] * 60 + m[1]
    return m[0] if m else None


def subs_chunks(s: str) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    cur, cur_start, cur_end = "", None, 0.0
    for block in s.split("\n\n"):
        text: list[str] = []
        start = None
        end = 0.0
        for line in block.splitlines():
            if "-->" in line:
                parts = line.split("-->")
                st, en = _parse_ts(parts[0]), _parse_ts(parts[-1])
                if st is not None and en is not None:
                    start = start if start is not None else st
                    end = en
            elif start is not None and line.strip():
                text.append(line.strip())
        if start is None or not text:
            continue
        t = "\n".join(text)
        if cur and len(cur) + len(t) > SUBS_CHUNK_CHARS:
            out.append((cur_start, cur_end, cur))
            cur, cur_start = "", None
        if cur:
            cur += "\n"
        cur += t
        cur_start = cur_start if cur_start is not None else start
        cur_end = end
    if cur:
        out.append((cur_start, cur_end, cur))
    return out


def wav16_mono(path: str) -> np.ndarray:
    """16k mono f32. Direct wave read when already 16k mono; else ffmpeg."""
    e = ext_of(path)
    if e == "wav":
        with wave.open(path, "rb") as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            sw_ok = sw == 2  # only 16-bit pcm reads as int16; anything else → ffmpeg
            if sw_ok:
                data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
            else:
                data = None
        if data is not None and sr == SR and ch == 1:
            return data
    import shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not installed — needed to decode audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=cache_dir()) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
             "-acodec", "pcm_s16le", tmp],
            check=True, capture_output=True, timeout=1200,
        )
        with wave.open(tmp, "rb") as w:
            n = w.getnframes()
            data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        return data
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def thumb256(path: str) -> bytes | None:
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((256, 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def cache_dir() -> str:
    d = os.path.expanduser("~/.cache/mediarag")
    os.makedirs(d, exist_ok=True)
    return d


# Domain stamps (path prefix → lane label). Empty by default for public installs;
# deployments override via MEDIARAG_DOMAIN_STAMPS — JSON list of ["<path prefix>", "<label>"],
# JSON string or path to a JSON file (example: [["/Users/me/docs/finance","finance"]]).
_AGENT_STAMPS: tuple = ()


def _load_stamps() -> tuple:
    raw = os.environ.get("MEDIARAG_DOMAIN_STAMPS", "")
    if not raw:
        return ()
    import json
    if os.path.isfile(raw):
        raw = open(raw).read()
    try:
        return tuple(tuple(x) for x in json.loads(raw))
    except Exception:
        return ()


def agent_of(path: str) -> str:
    stamps = _AGENT_STAMPS or _load_stamps()
    for prefix, agent in stamps:
        p = os.path.expanduser(prefix)
        if path.startswith(p + os.sep) or path == p:
            return agent
    return ""


def _fingerprint(path: str) -> str:
    """mtime+size signature for incremental skip (no content reads)."""
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime_ns)}:{st.st_size}"
    except OSError:
        return ""


def index_file(c, path: str, incremental: bool = True) -> tuple[int, list[dict]]:
    """Index one file into the corpus. Returns (rows_added, skipped[entries]);
    rows_added == -1 means unchanged-since-last-ingest (incremental skip)."""
    kind = kind_of(path)
    if kind is None:
        return 0, [{"uri": path, "code": "unsupported_ext"}]
    uri = os.path.abspath(path)
    if incremental:
        fp = _fingerprint(path)
        row = c.execute("SELECT fp FROM fingerprints WHERE uri = ?", (uri,)).fetchone()
        if row and fp and row[0] == fp:
            return -1, []  # unchanged since last ingest — skip, no re-embed
    rows: list[dict] = []
    try:
        e = ext_of(path)
        if e in SUBS_EXT or e in TEXT_EXT:
            size = os.path.getsize(path)
            if size > 50 * 1024 * 1024:  # ponytail: plain-text >50MB = OOM bait; skip, don't read
                return 0, [{"uri": path, "code": "too_large", "size": size}]
        if e in SUBS_EXT:
            with open(path, encoding="utf-8", errors="replace") as f:
                chunks = subs_chunks(f.read())
            if not chunks:
                return 0, [{"uri": path, "code": "empty"}]
            for s, e2, text in chunks:
                rows.append({"uri": uri, "kind": "text", "span_start": s, "span_end": e2,
                             "snippet": text, "vec": embed_doc(text), "thumb": None})
        elif e in TEXT_EXT or e in DOC_PANDOC_EXT or e in DOC_PDF_EXT:
            if e in TEXT_EXT:
                with open(path, encoding="utf-8", errors="replace") as f:
                    raw = f.read()
            else:
                raw = doc_text(path)
            if len(raw) < MIN_CONTENT_CHARS:
                return 0, [{"uri": path, "code": "empty"}]  # rag parity: min_content_chars
            chunks = text_chunks(raw)
            if not chunks:
                return 0, [{"uri": path, "code": "empty"}]
            for s, e2, text in chunks:
                rows.append({"uri": uri, "kind": "text", "span_start": s, "span_end": e2,
                             "snippet": text, "vec": embed_doc(text), "thumb": None})
        elif kind == "image":
            thumb = thumb256(path)
            if thumb is None:
                return 0, [{"uri": path, "code": "corrupt_format"}]
            rows.append({"uri": uri, "kind": "image", "span_start": 0.0, "span_end": 0.0,
                         "snippet": os.path.basename(path), "vec": embed_image(path),
                         "thumb": thumb})
        else:
            data = wav16_mono(path)
            if len(data) < SR // 10:
                return 0, [{"uri": path, "code": "empty"}]
            base = os.path.basename(path)
            for i in range(0, len(data), AUDIO_CHUNK):
                seg = data[i:i + AUDIO_CHUNK]
                if len(seg) < SR // 10:
                    break
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=cache_dir()) as f:
                    tmp = f.name
                try:
                    with wave.open(tmp, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(SR)
                        w.writeframes((np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes())
                    start = i / SR
                    end = start + len(seg) / SR
                    rows.append({"uri": uri, "kind": "audio", "span_start": start, "span_end": end,
                                 "snippet": f"{base} [{start:.1f}-{end:.1f}s]",
                                 "vec": embed_audio(tmp), "thumb": None})
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
    except Exception as ex:
        code = "corrupt_format" if "decode" in str(ex).lower() or "wave" in str(ex).lower() else "read_error"
        return 0, [{"uri": path, "code": code, "detail": str(ex)[:200]}]
    if not rows:
        return 0, [{"uri": path, "code": "empty"}]
    # Label row: filename + parent dir as its own embedded text row, so exact
    # producer-label lookups ("C6", "narration_take3") resolve in BOTH lanes
    # (dogfood structural miss). span_start=-1 avoids collision w/ content spans.
    label_rows_kind = rows[0]["kind"] if rows else "text"  # label rides the file's kind: kind=audio filters include it
    label = f"{os.path.basename(path)} — {os.path.basename(os.path.dirname(path))}"
    rows.append({"uri": uri, "kind": label_rows_kind, "span_start": -1.0, "span_end": -1.0,
                 "snippet": label, "vec": embed_doc(label), "thumb": None})
    agent = agent_of(uri)
    for r in rows:
        r["agent"] = agent
    store.upsert_uri(c, uri, rows, fp=_fingerprint(path))
    return len(rows), []


def index_dir(c, d: str, incremental: bool = True) -> dict:
    """Walk d, index every supported file. Per-file errors collected, never fatal."""
    processed = skipped = added = unchanged = 0
    for dirpath, dirnames, filenames in os.walk(d, followlinks=False):  # explicit: no symlink loops
        dirnames[:] = sorted(x for x in dirnames if not x.startswith("."))
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            if kind_of(p) is None:
                continue
            if excluded(p):
                continue
            n, skip = index_file(c, p, incremental=incremental)
            if n == -1:
                unchanged += 1
            elif n:
                added += n
                processed += 1
            elif skip:
                skipped += 1
    return {"processed": processed, "added": added, "skipped": skipped, "unchanged": unchanged}