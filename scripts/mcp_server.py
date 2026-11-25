"""MediaRAG MCP stdio server — Agent Plugins v1 transport.

The v1 client spawns this process per session and injects PLUGIN_ROOT/PLUGIN_DATA;
tools dispatch into the same core tool functions the native plugin toolset registers.
Keyless except the optional Raindrop and Notion adapters, which read
RAINDROP_API_TOKEN / NOTION_TOKEN from the environment the client injects.

Dual-era MCP (spec 2026-07-28): requests carrying a modern per-request
_meta.io.modelcontextprotocol/protocolVersion are served statelessly per that
revision (server/discover, -32022 version rejection, resultType on results);
requests without one get the legacy initialize handshake (<= 2025-11-25
revisions), which is what Agent Plugins v1 hosts and current MCP clients speak.
"""
import json
import os
import sys

_server_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_server_dir)
sys.path.insert(0, _repo_root)

SERVER_INFO = {"name": "mediarag", "version": "0.1.23"}
_MODERN = "2026-07-28"
_LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
_VER_KEY = "io.modelcontextprotocol/protocolVersion"


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
          "kind": {"type": "string", "enum": ["text", "image", "audio", "video"]},
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


def _requested_version(req):
    meta = (req.get("params") or {}).get("_meta") or {}
    return meta.get(_VER_KEY)


def _reject_version(rid, ver):
    # UnsupportedProtocolVersionError: both data fields are required by the schema.
    _write({"jsonrpc": "2.0", "id": rid, "error": {"code": -32022,
        "message": "Unsupported protocol version",
        "data": {"requested": ver, "supported": [_MODERN]}}})


def _serve(method, rid, req, schemas, handlers, modern):
    """One request in either era; `modern` only adds the resultType/cacheScope
    fields the 2026-07-28 schema mandates — legacy responses stay byte-identical
    to what shipped clients already parse."""
    era_extra = {"resultType": "complete", "cacheScope": "public", "ttlMs": 0} if modern else {}
    if method == "initialize":
        requested = (req.get("params") or {}).get("protocolVersion")
        reply = requested if requested in _LEGACY_VERSIONS else "2024-11-05"
        _write({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": reply, "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        _write({"jsonrpc": "2.0", "id": rid, "result": {"tools": schemas, **era_extra}})
    elif method == "tools/call":
        params = req.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in handlers or not isinstance(args, dict):
            _write({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601,
                "message": f"unknown tool: {name!r}"}})
            return
        try:
            result = handlers[name](**args)
            _write({"jsonrpc": "2.0", "id": rid, "result": {"content": [
                {"type": "text", "text": result if isinstance(result, str) else json.dumps(result)}],
                **era_extra}})
        except Exception as e:
            _write({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": f"error: {e}"}], "isError": True,
                **era_extra}})
    elif method == "ping":
        _write({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif rid is not None:
        _write({"jsonrpc": "2.0", "id": rid, "error": {
            "code": -32601, "message": f"Method not found: {method}"}})


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
        ver = _requested_version(req)

        if method == "server/discover":
            # The stdio backward-compat probe: an unversioned discover still
            # gets answered — the result itself tells the client what we speak.
            if ver is not None and ver != _MODERN:
                if rid is not None:
                    _reject_version(rid, ver)
            elif rid is not None:
                _write({"jsonrpc": "2.0", "id": rid, "result": {
                    "resultType": "complete", "supportedVersions": [_MODERN],
                    "capabilities": {"tools": {}},
                    "_meta": {"io.modelcontextprotocol/serverInfo": SERVER_INFO},
                    "instructions": "Local multimodal corpus memory: search and "
                                    "ingest documents, images, audio and video.",
                    "cacheScope": "public", "ttlMs": 3600000}})
        elif ver is not None and ver != _MODERN:
            if rid is not None:
                _reject_version(rid, ver)
        else:
            _serve(method, rid, req, schemas, handlers, modern=ver is not None)


if __name__ == "__main__":
    main()
