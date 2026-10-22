#!/usr/bin/env bash
# Local benchmark harness. Not the install path.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 tests/bench_engine.py
