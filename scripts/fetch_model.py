#!/usr/bin/env python3
"""One-time EG2 q4 download into the Hugging Face cache. About 480MB."""
from huggingface_hub import snapshot_download

snapshot_download(
    "onnx-community/embeddinggemma-2-ONNX",
    allow_patterns=[
        "*.json",
        "chat_template.jinja",
        "*.md",
        "onnx/*_q4.onnx",
        "onnx/*_q4.onnx_data",
    ],
)
print("model ready")
