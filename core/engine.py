"""MediaRAG plugin engine: EG2 towers via the HF processor + onnxruntime.

Mirrors scripts/export_golden.py exactly (the validated reference stack).
Parity gate: tests/bench_engine.py — cosine >= 0.999 vs goldens, all towers.

Memory law (live-verified): variable-shape audio/image workloads grow ort
arenas unbounded with the default memory pattern — every session sets
enable_mem_pattern=False (12 GB + 27 GB swap observed without it).
"""
from __future__ import annotations

import glob
import os
import sys
import wave
from functools import lru_cache

import numpy as np
import onnxruntime as ort

sys_path = os.environ.get("MEDIARAG_PYLIBS2") or os.path.expanduser("~/pylibs2")
if os.path.isdir(sys_path):
    sys.path.insert(0, sys_path)  # always first: venv copies of hub/older deps shadow it otherwise

from transformers import EmbeddingGemma2Processor  # noqa: E402
from PIL import Image  # noqa: E402

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
REPO = "onnx-community/embeddinggemma-2-ONNX"
MODEL_FILES = {
    "trunk": "model_q4.onnx",
    "vision": "vision_encoder_q4.onnx",
    "audio": "audio_encoder_q4.onnx",
}


def snap_dir() -> str:
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--onnx-community--embeddinggemma-2-ONNX/snapshots/*/onnx"
        )
    )
    if not hits:
        raise RuntimeError("embedding model not downloaded — run the model bootstrap")
    return sorted(hits)[-1]


def _sess(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.enable_mem_pattern = False  # memory law
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _processor():
    return EmbeddingGemma2Processor.from_pretrained(REPO)


@lru_cache(maxsize=4)
def _trunk() -> ort.InferenceSession:
    return _sess(os.path.join(snap_dir(), MODEL_FILES["trunk"]))


@lru_cache(maxsize=2)
def _vis() -> ort.InferenceSession:
    return _sess(os.path.join(snap_dir(), MODEL_FILES["vision"]))


@lru_cache(maxsize=2)
def _aud() -> ort.InferenceSession:
    return _sess(os.path.join(snap_dir(), MODEL_FILES["audio"]))


Z = np.zeros((0, 512), np.float32)


def _finish(feeds: dict) -> np.ndarray:
    v = np.asarray(_trunk().run(None, feeds)[1], dtype=np.float32).reshape(-1)  # [1] = sentence_embedding
    return v / np.linalg.norm(v)


def embed_query(q: str) -> np.ndarray:
    q = (QUERY_PREFIX + q.strip()).strip()
    p = _processor()(text=[q], return_tensors="np")
    return _finish({"input_ids": p["input_ids"].astype(np.int64),
                    "attention_mask": p["attention_mask"].astype(np.int64),
                    "image_features": Z, "video_features": Z, "audio_features": Z})


def embed_doc(raw: str) -> np.ndarray:
    raw = (DOC_PREFIX + raw.strip()).strip()
    if raw == DOC_PREFIX.strip():
        raise ValueError("empty text chunk")
    p = _processor()(text=[raw], return_tensors="np")
    return _finish({"input_ids": p["input_ids"].astype(np.int64),
                    "attention_mask": p["attention_mask"].astype(np.int64),
                    "image_features": Z, "video_features": Z, "audio_features": Z})


def embed_plain(raw: str) -> np.ndarray:
    """verbatim text, no prompt template (caption/label lane)."""
    raw = raw.strip()
    if not raw:
        raise ValueError("empty text chunk")
    p = _processor()(text=[raw], return_tensors="np")
    return _finish({"input_ids": p["input_ids"].astype(np.int64),
                    "attention_mask": p["attention_mask"].astype(np.int64),
                    "image_features": Z, "video_features": Z, "audio_features": Z})


def embed_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    # ponytail: oversized images blow up the soft-token patch grid → onnxruntime
    # arena abort (observed: 14 TB single-buffer request kills the process).
    # Cap at 64 MP (8192^2): EG2 resizes to ≤ 560 px patches anyway — above this
    # cap the pixels are pure waste. Upgrade path: pre-resize instead of reject.
    if img.width * img.height > 64_000_000 or img.width > 16_384 or img.height > 16_384:
        raise ValueError(f"image too large to embed: {img.size}")
    p = _processor()(images=[img], return_tensors="np")
    enc = _vis().run(None, {"pixel_values": p["pixel_values"],
                            "pixel_position_ids": p["image_position_ids"].astype(np.int64)})[0]
    n = int((p["input_ids"][0] == _processor().image_token_id).sum())
    assert n == enc.shape[0], f"image token mismatch: {n} != {enc.shape[0]}"
    return _finish({"input_ids": p["input_ids"].astype(np.int64),
                    "attention_mask": p["attention_mask"].astype(np.int64),
                    "image_features": enc.astype(np.float32),
                    "video_features": Z, "audio_features": Z})


def embed_audio(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    p = _processor()(audio=[a], sampling_rate=16000, return_tensors="np")
    enc = _aud().run(None, {"input_features": p["input_features"].astype(np.float32),
                            "input_features_mask": p["input_features_mask"]})[0]
    n = int((p["input_ids"][0] == _processor().audio_token_id).sum())
    assert n == enc.shape[0], f"audio token mismatch: {n} != {enc.shape[0]}"
    return _finish({"input_ids": p["input_ids"].astype(np.int64),
                    "attention_mask": p["attention_mask"].astype(np.int64),
                    "image_features": Z, "video_features": Z,
                    "audio_features": enc.astype(np.float32)})