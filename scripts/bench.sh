#!/usr/bin/env bash
# MediaRAG bench: engine vs goldens (12 vectors, gate cosine >= 0.999).
# Env recipe (verified 2026-10-14, 12/12 at cos 1.000000):
#   - onnxruntime/numpy/PIL/tokenizers live in a Hermes install-environment
#     venv: ~/.hermes/installs/<id>/environments/<id>/venv
#   - transformers 5.19 + the EG2 processor come from ~/pylibs2
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="$(ls -d "$HOME"/.hermes/installs/*/environments/*/venv 2>/dev/null | head -1)"
if [[ -z "${VENV:-}" ]]; then
  echo "ERROR: no hermes install-environment venv found" >&2
  exit 1
fi
export PYTHONPATH="$VENV/lib/python3.14/site-packages:$HOME/pylibs2"
PY="$(dirname "$(command -v python3)")/python3"
echo "python=$PY site=$VENV"
exec "$PY" tests/bench_engine.py