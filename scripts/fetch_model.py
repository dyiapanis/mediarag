#!/usr/bin/env python3
"""One-time EG2 q4 download into the Hugging Face cache. About 480MB."""
import os
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_pin import REPO, REVISION  # noqa: E402 — the pin lives with the engine

snapshot_download(
    REPO,
    revision=REVISION,  # pinned: the parity-locked snapshot
    allow_patterns=[
        "*.json",
        "chat_template.jinja",
        "*.md",
        "onnx/*_q4.onnx",
        "onnx/*_q4.onnx_data",
    ],
)
print("model ready")
