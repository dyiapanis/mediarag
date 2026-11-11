"""_materialize must be safe when two processes race on the same snapshot.

Regression for the unlocked copy/delete race: one process could rmtree a shard
directory while another was reading it.

Run: python tests/test_materialize_race.py   (no model needed, small fake shards)
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import engine  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


tmp = tempfile.mkdtemp(prefix="mediarag-race-")
try:
    src = os.path.join(tmp, "src")
    dst = os.path.join(tmp, "dst")
    os.makedirs(src)
    expected = {}
    for name in engine.MODEL_FILES.values():
        for fn in (name, name + "_data"):
            content = (fn.encode() * 97)[:4096]
            with open(os.path.join(src, fn), "wb") as fh:
                fh.write(content)
            expected[fn] = content

    for round_no in range(3):
        code = ("import sys; sys.path.insert(0, %r); from core import engine; "
                "print(engine._materialize(%r, %r))") % (ROOT, src, dst)
        procs = [subprocess.Popen([sys.executable, "-c", code],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for _ in range(2)]
        outs = [p.communicate(timeout=180) for p in procs]
        rcs = [p.returncode for p in procs]
        check(f"round {round_no}: both processes succeed", rcs == [0, 0],
              [o[1].decode()[-200:] for o in outs])
        check(f"round {round_no}: destination exists", os.path.isdir(dst))
        for fn, content in expected.items():
            path = os.path.join(dst, fn)
            check(f"round {round_no}: {fn} intact",
                  os.path.isfile(path) and open(path, "rb").read() == content)
        check(f"round {round_no}: no .tmp left", not os.path.isdir(dst + ".tmp"))
        check(f"round {round_no}: no .old left", not os.path.isdir(dst + ".old"))
        with open(os.path.join(dst, ".src")) as fh:
            check(f"round {round_no}: stamp matches source", fh.read() == src)
        shutil.rmtree(dst)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nALL MATERIALIZE RACE CHECKS PASS")
