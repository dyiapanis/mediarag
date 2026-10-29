"""MediaRAG MCP stdio server — Agent Plugins v1 transport.

The v1 client spawns this process per session and injects PLUGIN_ROOT/PLUGIN_DATA;
tools dispatch into the same core tool functions the native plugin toolset registers.
Keyless except the optional Raindrop adapter, which reads RAINDROP_API_TOKEN
from the environment the client injects.
"""
import json
import os
import sys

_server_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_server_dir)
sys.path.insert(0, _repo_root)

SERVER_INFO = {"name": "mediarag", "version": "0.1.7"}


def _bootstrap_env() -> bool:
    """Engine deps come from the install; verify them, say what to do if not.

    No sys.path editing and no process-wide HF_HUB_OFFLINE here: the engine asks
    Hugging Face for nothing once the weights are on disk (from_pretrained runs
    with local_files_only=True).
    """
    try:
        from core import engine  # noqa: F401
        return True
    except ImportError as ex:
        print(f"[mediarag] engine deps missing ({ex}) — pip install -r requirements.txt",
              file=sys.stderr)
        return False


def _table():
    """The 5-tool table: (name, description, handler, params, required)."""
    from core import tools as T
    return [
        ("mediarag_search",
         "Search documents and media by meaning: text, images, audio. Ranked hits "
         "with snippets, scores, spans, and image thumbnails.",
         T.tool_search,
         {"query": {"type": "string", "description": "natural-language query"},
          "k": {"type": "integer", "default": 5},
          "kind": {"type": "string", "enum": ["text", "image", "audio"]},
          "include_thumbnails": {"type": "boolean", "default": True},
          },
         ["query"]),
        ("mediarag_add_note",
         "Store a text note into the MediaRAG corpus (searchable afterwards).",
         T.tool_add_note,
         {"text": {"type": "string", "description": "note text"},
          },
         ["text"]),
        ("mediarag_ingest_dir",
         "Bulk-index a directory of documents and media (pdf, office, notes, "
         "subtitles, images, audio). Per-file errors are collected, never fatal.",
         T.tool_ingest_dir,
         {"path": {"type": "string"},
          },
         ["path"]),
        ("mediarag_add_source",
         "Add one source: local file/dir, http(s) URL, or adapter scheme URI "
         "(raindrop://, pcloud://, obsidian://, rclone://backend:path).",
         T.tool_add_source,
         {"source": {"type": "string"},
          },
         ["source"]),
        ("mediarag_status",
         "MediaRAG corpus status: row counts by kind, corpus root, adapters.",
         T.tool_status, {}, []),
    ]


def _read_message():
    """One JSON-RPC message per line (MCP stdio spec framing)."""
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 1 << 20:  # ponytail: 1MB line cap, no streaming JSON here
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    return None


def _write(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    if not _bootstrap_env():
        return 1
    table = _table()
    schemas = [{"name": n, "description": d,
                "inputSchema": {"type": "object", "properties": p, "required": r}}
               for n, d, _, p, r in table]
    handlers = {n: h for n, _, h, _, _ in table}

    while True:
        req = _read_message()
        if req is None:
            break
        method, rid = req.get("method"), req.get("id")
        if method == "initialize":
            _write({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _write({"jsonrpc": "2.0", "id": rid, "result": {"tools": schemas}})
        elif method == "tools/call":
            params = req.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in handlers or not isinstance(args, dict):
                _write({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601,
                    "message": f"unknown tool: {name!r}"}})
                continue
            try:
                result = handlers[name](**args)
                _write({"jsonrpc": "2.0", "id": rid, "result": {"content": [
                    {"type": "text", "text": result if isinstance(result, str) else json.dumps(result)}]}})
            except Exception as e:
                _write({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}], "isError": True}})
        elif method == "ping":
            _write({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif rid is not None:
            _write({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()