"""The single source of truth for the embedding model's pinned revision.

Both the downloader (scripts/fetch_model.py) and the runtime resolver
(core/engine.py) read these, so the weights we fetch and the weights we load
can never drift apart — previously the engine picked the newest snapshot
directory instead of the pinned one.
"""

REPO = "onnx-community/embeddinggemma-2-ONNX"
REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
CACHE_DIRNAME = "models--onnx-community--embeddinggemma-2-ONNX"
