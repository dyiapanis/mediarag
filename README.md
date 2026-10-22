# MediaRAG

Multimodal local search over documents and media, packaged as a Hermes
agent-plugin per the [agent-plugins.org](https://agent-plugins.org/) 1.0
standard.

It indexes documents (pdf, office files, notes, html) and media (images,
audio, video soundtracks, subtitles) into one local SQLite corpus, and you
search both by meaning from any MCP-capable agent. "Which file mentions the
landing" and "where is the clip where I talked about it" are the same kind of
question.

How it runs:

- In-gateway: native tools (`search`, `add_source`, `ingest_dir`, `add_note`,
  `status`) register once, so one gateway serves all agents and the embedding
  model loads once.
- Cross-client: `mcp.json` declares a stdio MCP server, so Claude Desktop,
  Cursor, and other Agent Plugins compatible clients load the same package.
- Local-only: no cloud services. The corpus lives at `<root>/.corpus/store.db`.

The text, image, and audio embedders are parity-locked against the reference
models: 12/12 golden vectors at cosine 1.000000.

## Features

- Hybrid retrieval: semantic vectors plus an FTS5/BM25 keyword lane, fused with
  reciprocal rank fusion. Exact-string lookups and meaning searches both work.
- Documents and media in one index: pdf via `pdftotext`, docx/odt/epub/rst/html/csv/rtf
  via `pandoc`, plus images, audio, and subtitles.
- Incremental re-indexing: unchanged files are skipped on an mtime+size
  fingerprint.
- Source adapters: external collections through `sources.d` executables (see
  below).

## Install

Python dependencies: `pip install -r requirements.txt`. The system tools are
only needed for the lanes that use them: `ffmpeg` for audio, `pandoc` for
documents, `pdftotext` for pdf.

The embedding weights are not downloaded on startup. Once, before first search:

```bash
python scripts/fetch_model.py
```

That is about 480MB into the Hugging Face cache. A sharded cache is copied once to
`~/.cache/mediarag/onnx-real` so onnxruntime can load it. After that everything
runs offline (`HF_HUB_OFFLINE=1`).

## What it touches

No telemetry. It shells out to `curl` (http sources), `ffmpeg`, `pandoc`,
`pdftotext`, and `rclone` when those lanes are used, and to whatever adapter
you drop in `sources.d/`. It reads the paths you pass plus the corpus at
`~/.mediarag` (or `MEDIARAG_ROOT`). The Raindrop adapter calls
`api.raindrop.io` with `RAINDROP_API_TOKEN` — optional; inside Hermes the token
is resolved through the profile secret scope, and a missing scope does not
inherit another profile's value.

To run it as a Hermes plugin, point the plugin loader at this directory (or at
the agent plugin catalog entry). The manifests (`mcp.json`, `plugin.yaml`,
`plugin.json`) describe both the stdio MCP server (`scripts/mcp_server.py`,
spawned by the client) and the in-gateway native tools. Neither mode opens a
network listener.

## Configuration

Everything is optional and set through the environment:

- `mediarag-config.yaml` — named corpus config in the plugin dir (`root:` key); wins over env and the default (see `mediarag-config.example.yaml`)
- `MEDIARAG_ROOT` — corpus root (default: the host profile's data dir)
- `MEDIARAG_PYLIBS2` — extra site-packages path for the engine dependencies
- `RAINDROP_API_TOKEN`, `PCLOUD_MOUNT`, `OBSIDIAN_VAULT` — adapter inputs

## Adapters

`add_source` accepts `http(s)://` URLs (fetched with curl, capped at 256MB)
and source-scheme URIs (`<scheme>://...`) handled by adapters. Four ship in the
box:

- `rclone://remote:path` — any [rclone](https://rclone.org/) remote. Files copy
  into a persistent cache at `~/.cache/mediarag/rclone` and index from there.
- `pcloud://` — walks a locally mounted pCloud drive (`PCLOUD_MOUNT` env).
- `obsidian://` — an Obsidian vault; skips `.obsidian/`, `.trash/`, `.git/`.
- `raindrop://` — Raindrop.io bookmarks through the REST API
  (`RAINDROP_API_TOKEN` env).

To write your own adapter, drop an executable at `sources.d/<scheme>` that
accepts `fetch <rest>` and prints one JSON object per line, either
`{"uri","title","text"}` or `{"uri","path"}`. `status` reports the adapters it
found; an unknown scheme fails with the list of what is available.

## Development

Tests are direct-run scripts, not pytest-collectable: `python tests/test_ingest_units.py`,
`python tests/test_engine_deref.py`, `python tests/test_parity_slices.py`,
`python tests/test_config_root.py`. The fixtures under `tests/fixtures/` are
synthetic and deterministic, and nothing contacts the network. `tests/bench_engine.py` and
`scripts/bench.sh` are the local benchmark harness.

## License

MIT. For security reports, see `SECURITY.md`.

## Acknowledgements

This project was developed with AI assistance using [Hermes Agent](https://github.com/NousResearch/hermes-agent).
