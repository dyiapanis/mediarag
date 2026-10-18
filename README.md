# MediaRAG

Multimodal local asset memory for content creators — a Hermes agent-plugin
package per the [agent-plugins.org](https://agent-plugins.org/) 1.0 standard.

Index your media (images, audio, video soundtracks, notes, subtitles) into a
local SQLite corpus and search it by meaning with any MCP-capable agent.

- **In-gateway**: registers native tools (`search`, `add_source`, `ingest_dir`,
  `add_note`, `status`) once — a single gateway serves all agents, the embedding
  model loads once.
- **Cross-client**: `mcp.json` declares an MCP server entry, so Claude Desktop,
  Cursor and other Agent-Plugins-compatible clients load the same package.
- **Local-only**: no cloud services; the corpus lives at `<root>/.corpus/store.db`.

Status: v1 agent-plugin (python). Text/image/audio towers parity-locked at
12/12 golden vectors (cos 1.000000). See pCloud `MediaRAG-scope.md` for the
build contract.

Fleet-corpus edition: **hybrid retrieval** (semantic + FTS5/BM25 keyword lane,
RRF-fused), **doc ingest** (pdf via `pdftotext`; docx/odt/epub/rst/html/csv/rtf
via `pandoc`), **per-agent domain stamps** (search param `agent`), and
**incremental re-indexing** (unchanged files skip by mtime+size fingerprint).
System tools expected when their lane is used: `ffmpeg` (audio), `pandoc`
(documents), `pdftotext` (pdf). First use downloads the EG2 q4 ONNX weights
(~482MB) to the HF cache. Domain stamps are opt-in per deployment:
`MEDIARAG_DOMAIN_STAMPS` = JSON list (or path to a JSON file) of
`["<path prefix>", "<label>"]` pairs; unmatched paths stay in the shared lane.

## Adapters

`add_source` accepts `http(s)://` URLs (curl, ≤256MB) and source-scheme URIs
(`<scheme>://...`) handled by adapters. Four ship bundled (all optional-lane
system deps):

- `rclone://remote:path` — any [rclone](https://rclone.org/) remote; files copy
  into a persistent cache (`~/.cache/mediarag/rclone`) then index natively
- `pcloud://` — walks a locally-mounted pCloud drive (`PCLOUD_MOUNT` env)
- `obsidian://` — Obsidian vault; skips `.obsidian/`, `.trash/`, `.git/`
- `raindrop://` — Raindrop.io bookmarks via REST (`RAINDROP_API_TOKEN` env)

Custom adapters: drop an executable in `sources.d/<scheme>` that accepts
`fetch <rest>` and emits JSON lines `{"uri","title","text"}` or
`{"uri","path"}` on stdout. `status` lists discovered adapters; unknown
schemes fail with the available list.