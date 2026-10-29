"""MediaRAG plugin entry — agent-plugins 1.0, loaded as hermes_plugins.<slug>.

register(ctx) is the real loader contract: ctx.register_tool(name, toolset,
schema={"type":"object","parameters":…}, handler, description). Handlers
absorb dispatch kwargs themselves (**_x — Hermes passes task_id + extras).

Env contract: register() checks that onnxruntime, numpy, and transformers
import, and refuses to register tools when they do not. Install them with
`pip install -r requirements.txt`. The model snapshot is loaded from the
Hugging Face cache and is not downloaded on startup.
"""
from __future__ import annotations

import os
import sys

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

def _config_root() -> str:
    """Corpus root from mediarag-config.yaml in the plugin dir (named config, rag-config.yaml pattern).
    # ponytail: flat `root:` line parse — no yaml dependency on the stdio spawn path;
    # switch to yaml.safe_load if the config ever grows beyond one key.
    """
    path = os.path.join(_PLUGIN_DIR, "mediarag-config.yaml")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("root:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


TOOL_ROOT = _config_root() or os.environ.get("MEDIARAG_ROOT") or os.path.expanduser("~/.mediarag")

PYVER = f"python{sys.version_info.major}.{sys.version_info.minor}"

BOOTSTRAP_HINT = (
    "MediaRAG engine deps missing — pip install -r requirements.txt "
    "from the plugin directory"
)


def _engine_importable() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import numpy  # noqa: F401
        import transformers  # noqa: F401
        # The processor chain constructs a video processor needing BOTH torch and
        # torchvision; without them every embed raises ImportError at call time
        # while the plugin still advertises five working tools.
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def register(ctx) -> None:
    """MediaRAG native tool registration (Hermes loader contract)."""
    # Deps arrive with the plugin install (python_dependencies / requirements.txt).
    # Nothing here edits sys.path or the process environment: this runs inside the
    # Hermes gateway, where either would leak into every other plugin.
    if not _engine_importable():
        print(f"[mediarag] {BOOTSTRAP_HINT}")
        return

    import json as _json

    from .core import tools as _tools

    def _json_result(handler):
        """Registry dispatch contract: handler receives ONE positional arg (the
        args dict, task_id-style extras included) and returns a str (dict
        results are rejected — the workspace plugin's lifetime bug)."""
        def wrapped(args=None, **kwargs):
            a = dict(args or {})
            a.update(kwargs)
            return _json.dumps(handler(**a))
        return wrapped

    def reg(name, desc, handler, params, required):
        ctx.register_tool(
            name=name,
            toolset="mediarag",
            schema={"type": "object", "parameters": {
                "type": "object", "properties": params, "required": required}},
            handler=_json_result(handler),
            check_fn=lambda: True,
            description=desc,
        )

    reg(
        "mediarag_search",
        "Search documents and media (hybrid: semantic + keyword/BM25): text, images, "
        "audio. Ranked hits with snippets, scores, spans, and image thumbnails.",
        _tools.tool_search,
        {
            "query": {"type": "string", "description": "natural-language query (or keyword terms)"},
            "k": {"type": "integer", "default": 5},
            "kind": {"type": "string", "enum": ["text", "image", "audio"]},
            "include_thumbnails": {"type": "boolean", "default": True},
        },
        ["query"],
    )
    reg(
        "mediarag_add_note",
        "Store a text note into the MediaRAG corpus (searchable afterwards).",
        _tools.tool_add_note,
        {"text": {"type": "string", "description": "note text"},
         },
        ["text"],
    )
    reg(
        "mediarag_ingest_dir",
        "Bulk-index a directory of documents and media (pdf, office, notes, "
        "subtitles, images, audio). Per-file errors are collected, never fatal.",
        _tools.tool_ingest_dir,
        {"path": {"type": "string"},
         },
        ["path"],
    )
    reg(
        "mediarag_add_source",
        "Add one source: local file/dir, http(s) URL, or adapter scheme URI "
        "(raindrop://, pcloud://, obsidian://, rclone://backend:path).",
        _tools.tool_add_source,
        {"source": {"type": "string"},
         },
        ["source"],
    )
    reg(
        "mediarag_status",
        "MediaRAG corpus status: row counts by kind, corpus root, adapters.",
        _tools.tool_status,
        {},
        [],
    )
    print(f"[mediarag] 5 tools registered (corpus root: {TOOL_ROOT})")