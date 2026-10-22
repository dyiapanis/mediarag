#!/usr/bin/env python3
"""Sharded HF blobs must materialize to real files before onnxruntime sees them.

Direct-run: python tests/test_engine_deref.py (exit 0 = all pass). No model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import engine  # noqa: E402

PASSED = FAILED = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"PASS {name}")
    else:
        FAILED += 1
        print(f"FAIL {name}  {detail}")


def _link(onnx, name, target):
    os.symlink(target, os.path.join(onnx, name))


def main() -> int:
    import tempfile
    tmp = tempfile.mkdtemp(prefix="deref-")
    try:
        flat = os.path.join(tmp, "flat")
        os.makedirs(flat)
        blob = os.path.join(tmp, "blobs")
        os.makedirs(blob)
        g = os.path.join(blob, "model_q4.onnx")
        d = os.path.join(blob, "model_q4.onnx_data")
        open(g, "wb").write(b"graph")
        open(d, "wb").write(b"data")
        _link(flat, "model_q4.onnx", g)
        _link(flat, "model_q4.onnx_data", d)
        check("flat blobs dir does not escape", not engine._external_escapes(flat))

        real = os.path.join(tmp, "real")
        os.makedirs(real)
        open(os.path.join(real, "model_q4.onnx"), "wb").write(b"g")
        open(os.path.join(real, "model_q4.onnx_data"), "wb").write(b"d")
        check("real files do not escape", not engine._external_escapes(real))

        sharded = os.path.join(tmp, "sharded")
        os.makedirs(sharded)
        b1 = os.path.join(tmp, "blobs", "d7")
        b2 = os.path.join(tmp, "blobs", "23")
        os.makedirs(b1)
        os.makedirs(b2)
        sg = os.path.join(b1, "model_q4.onnx")
        sd = os.path.join(b2, "model_q4.onnx_data")
        open(sg, "wb").write(b"graph-bytes")
        open(sd, "wb").write(b"data-bytes")
        _link(sharded, "model_q4.onnx", sg)
        _link(sharded, "model_q4.onnx_data", sd)
        check("sharded blobs escape", engine._external_escapes(sharded))

        out = os.path.join(tmp, "out")
        dst = engine._materialize(sharded, dest=out)
        gp = os.path.join(dst, "model_q4.onnx")
        dp = os.path.join(dst, "model_q4.onnx_data")
        check("materialize writes real files", not os.path.islink(gp) and not os.path.islink(dp))
        check("materialize follows the symlink", open(dp, "rb").read() == b"data-bytes")
        check("second call is a no-op", engine._materialize(sharded, dest=out) == dst)
        check("stamp records the source", open(os.path.join(dst, ".src")).read() == sharded)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_v = [l.split(":", 1)[1].strip() for l in open(os.path.join(root, "plugin.yaml")) if l.startswith("version:")][0]
        import json
        json_v = json.load(open(os.path.join(root, "plugin.json")))["version"]
        srv = open(os.path.join(root, "scripts", "mcp_server.py")).read()
        check("version surfaces match", yaml_v == json_v and f'"version": "{yaml_v}"' in srv, f"{yaml_v} {json_v}")
        from core import tools as _tools
        os.environ.pop("RAINDROP_API_TOKEN", None)
        check("unset raindrop token is not inherited", "RAINDROP_API_TOKEN" not in _tools._adapter_env())
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
