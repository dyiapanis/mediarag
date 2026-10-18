# Security

Report vulnerabilities to d@yiapanis.co. Do not file public issues for security details.

- Tool surface: local-only. No cloud calls after the one-time model download (HF cache, `HF_HUB_OFFLINE=1` at runtime).
- External converters (`pandoc`, `pdftotext`, `ffmpeg`) run with timeouts and no shell expansion; add_source enforces size/time caps.
- MCP surface is stdio (client-spawned); there is no network listener.
