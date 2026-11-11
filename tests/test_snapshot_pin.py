"""Snapshot resolution must use the exact pinned revision, not the newest directory.

Run: python tests/test_snapshot_pin.py   (no model, no network)
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import engine  # noqa: E402
from core.model_pin import CACHE_DIRNAME, REVISION  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


tmp = tempfile.mkdtemp(prefix="mediarag-pin-")
old = os.environ.get("HF_HUB_CACHE")
os.environ["HF_HUB_CACHE"] = tmp
try:
    base = os.path.join(tmp, CACHE_DIRNAME, "snapshots")
    pinned = os.path.join(base, REVISION)
    decoy = os.path.join(base, "ffffffffffffffffffffffffffffffffffffffff")
    for d in (pinned, decoy):
        os.makedirs(os.path.join(d, "onnx"), exist_ok=True)

    resolved = engine._snapshot_dir()
    check("pinned revision wins over a newer directory", resolved == pinned, resolved)
    check("decoy is not chosen", resolved != decoy)

    shutil.rmtree(pinned)
    try:
        engine._snapshot_dir()
        check("missing pinned revision raises", False, "no error raised")
    except RuntimeError as e:
        check("missing pinned revision raises", "fetch_model.py" in str(e), str(e))
    # A decoy on its own is still not good enough — no silent fallback.
    check("decoy alone still raises", not os.path.isdir(pinned))
finally:
    if old is None:
        os.environ.pop("HF_HUB_CACHE", None)
    else:
        os.environ["HF_HUB_CACHE"] = old
    shutil.rmtree(tmp, ignore_errors=True)

# $VAR forms must resolve the way huggingface_hub resolves them
saved = {k: os.environ.get(k) for k in ("HF_HUB_CACHE", "HF_HOME", "MEDIARAG_TEST_BASE")}
try:
    os.environ["MEDIARAG_TEST_BASE"] = "/opt/models"
    os.environ["HF_HUB_CACHE"] = "$MEDIARAG_TEST_BASE/hub"
    os.environ.pop("HF_HOME", None)
    check("HF_HUB_CACHE expands $VAR", engine._hub_cache() == "/opt/models/hub",
          engine._hub_cache())
    os.environ.pop("HF_HUB_CACHE", None)
    os.environ["HF_HOME"] = "$MEDIARAG_TEST_BASE/hf"
    check("HF_HOME expands $VAR", engine._hub_cache() == os.path.join("/opt/models/hf", "hub"),
          engine._hub_cache())
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print("\nALL SNAPSHOT PIN CHECKS PASS")
