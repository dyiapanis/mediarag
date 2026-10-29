# MediaRAG

![MediaRAG — one semantic index over documents, images and audio](assets/banner.jpg)

Multimodal local search over documents and media, packaged as a Hermes
agent-plugin per the [agent-plugins.org](https://agent-plugins.org/) 1.0
standard.

It indexes documents (pdf, office files, notes, html) and media (images, audio,
subtitles) into one local SQLite corpus, and you search both by meaning from any
MCP-capable agent. Video is not an ingest format: pull the soundtrack out first
(`ffmpeg -i clip.mp4 -vn -ac 1 -ar 16000 soundtrack.wav`) and index that — the
audio tower then makes the clip findable by what was said in it. "Which file mentions the
landing" and "where is the clip where I talked about it" are the same kind of
question.

How it runs:

- In-gateway: five native tools (`mediarag_search`, `mediarag_add_source`,
  `mediarag_ingest_dir`, `mediarag_add_note`, `mediarag_status`) register once,
  so one gateway serves all agents and the embedding model loads once.
- Cross-client: `mcp.json` declares a stdio MCP server, so Claude Desktop,
  Cursor, and other Agent Plugins compatible clients load the same package.
- Local-only: no cloud services. The corpus lives at `<root>/.corpus/store.db`.

Documents and media are embedded by the same model into the same vector space
(see [Embedding model](#embedding-model)), which is why one index answers both
kinds of question.

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

Runs on Linux and Windows. On Windows those system tools are not preinstalled —
`winget install Gyan.FFmpeg JohnMacFarlane.Pandoc oschwartz10612.Poppler` (or the
builds of your choice) and make sure they are on `PATH`. The corpus cache lives
at `~\.cache\mediarag`; adapters are python scripts and are run through the
current interpreter there, so they do not need the executable bit or a shebang.

The embedding weights are not downloaded on startup. Once, before first search:

```bash
python scripts/fetch_model.py
```

That is about 285MB into the Hugging Face cache; what is in it, and why, is
covered under [Embedding model](#embedding-model). After the download everything
runs offline (`HF_HUB_OFFLINE=1`).

## Embedding model

One model embeds everything: [EmbeddingGemma
2](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) in its ONNX q4
build, on onnxruntime, on the CPU. It ships three towers — the text trunk, a
vision encoder and an audio encoder — and all three write into the same
768-dimensional space, so a pdf, a photograph and a clip of you talking about
both are directly comparable.

- **Snapshot** `daa72c51243991dfcaf9f9137d2c573d8f7790c0` of
  `onnx-community/embeddinggemma-2-ONNX`. `scripts/fetch_model.py` pulls only the
  q4 towers plus the processor configs, about 285MB, and never at startup. (A
  cache that has also seen other quantisations of this model will be larger —
  482MB on the machine this was written on — because those blobs are still there.)
- **768 dimensions**, L2-normalised, stored as f32 blobs (3,072 bytes per row).
  Retrieval is exact cosine over the whole corpus; there is no approximate index.
- **1024-token cap** on the text trunk. Attention is quadratic in sequence
  length, so tool input is sliced at that boundary rather than trusted.
- **Chunking**: documents and notes at 1,500 characters, subtitles at 600,
  audio in 11.2-second windows.
- If the Hugging Face cache stores the ONNX shards as symlinks, they are
  materialised once into `~/.cache/mediarag/onnx-real`: onnxruntime will not
  load external-data blobs out of a sharded cache.

The three embedders are parity-locked against the reference model — 12/12 golden
vectors at cosine 1.000000 (`tests/fixtures/golden.json`, enforced by
`tests/test_parity_slices.py`).

## What it touches

No telemetry. It shells out to `curl` (http sources), `ffmpeg`, `pandoc`,
`pdftotext`, and `rclone` when those lanes are used, and to whatever adapter
you drop in `sources.d/`. It reads the paths you pass plus the corpus at `root`
(`mediarag-config.yaml` `root:`, else `MEDIARAG_ROOT`, else `~/.mediarag`). The Raindrop adapter calls
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
- `MEDIARAG_ROOT` — corpus root (default: `~/.mediarag`)
- `MEDIARAG_ALLOW_PRIVATE_URLS=1` — let `add_source` fetch loopback/private-network URLs (standalone; inside Hermes the host's `security.allow_private_urls` applies). Link-local and cloud-metadata addresses are always refused.
- `RAINDROP_API_TOKEN`, `PCLOUD_MOUNT`, `OBSIDIAN_VAULT` — adapter inputs

## Adapters

`add_source` accepts public `http(s)://` URLs (fetched with curl, capped at 256MB;
every redirect hop is checked and loopback/private/link-local/metadata hosts are refused)
and source-scheme URIs (`<scheme>://...`) handled by adapters. Five ship in the
box:

- `dropbox://` — Dropbox through the desktop client's local folder, or any mirror
  (`DROPBOX_DIR` env). Skips Dropbox's own `.dropbox/` bookkeeping.
- `rclone://remote:path` — any [rclone](https://rclone.org/) remote. Files copy
  into a persistent cache at `~/.cache/mediarag/rclone` and index from there.
- `pcloud://` — walks a locally mounted pCloud drive (`PCLOUD_MOUNT` env).
- `obsidian://` — an Obsidian vault; skips `.obsidian/`, `.trash/`, `.git/`.
- `raindrop://` — Raindrop.io bookmarks through the REST API
  (`RAINDROP_API_TOKEN` env).

To write your own adapter, drop an executable at `sources.d/<scheme>` under the
configured corpus root (`mediarag-config.yaml` / `MEDIARAG_ROOT` / `~/.mediarag`; a
per-call `root` argument never adds adapters) that
accepts `fetch <rest>` and prints one JSON object per line, either
`{"uri","title","text"}` or `{"uri","path"}`. `status` reports the adapters it
found; an unknown scheme fails with the list of what is available.

## Development

Tests are direct-run scripts, not pytest-collectable:

- `python tests/test_contract.py` — tool/store/adapter contracts (no model)
- `python tests/test_ingest_units.py` — chunking, subtitle timings, store integrity
- `python tests/test_engine_deref.py` — sharded-cache materialisation (no model)
- `python tests/test_config_root.py` — corpus-root precedence (no model)
- `python tests/test_parity_slices.py` — FTS lane, RRF fusion, doc bridge (needs the model)

The fixtures under `tests/fixtures/` are synthetic and deterministic, and nothing
contacts the network. `tests/bench_engine.py` and `scripts/bench.sh` are the
local benchmark harness. CI runs the model-free suites on Linux and Windows,
plus the full parity lane on Linux.

## License

MIT. For security reports, see `SECURITY.md`.

## Acknowledgements

This project was developed with AI assistance using [Hermes Agent](https://github.com/NousResearch/hermes-agent).
