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

# --- promotion failure must roll back, never leave no destination -----------
# Inject a failure on the SECOND rename (tmp -> dst) and prove the old tree is
# restored rather than dst vanishing with only .old/.tmp left on disk.
import time  # noqa: E402

rollback_root = tempfile.mkdtemp(prefix="mediarag-swap-")
real_rename, real_sleep = os.rename, time.sleep
try:
    dst = os.path.join(rollback_root, "dst")
    tmpdir = os.path.join(rollback_root, "dst.tmp")
    os.makedirs(dst)
    os.makedirs(tmpdir)
    open(os.path.join(dst, "old.txt"), "w").write("previous good tree")
    open(os.path.join(tmpdir, "new.txt"), "w").write("replacement")

    def failing_rename(a, b):
        if a == tmpdir:
            raise OSError("injected promotion failure")
        return real_rename(a, b)

    os.rename = failing_rename
    time.sleep = lambda s: None  # skip the retry backoff; the failure is permanent here
    try:
        engine._swap_dir(tmpdir, dst)
        check("failed promotion raises", False, "no error raised")
    except OSError:
        check("failed promotion raises", True)
    check("destination still present after failure", os.path.isdir(dst))
    check("previous content rolled back",
          open(os.path.join(dst, "old.txt")).read() == "previous good tree")
    check("old tree is not stranded as .old", not os.path.isdir(dst + ".old"))

    os.rename, time.sleep = real_rename, real_sleep
    engine._swap_dir(tmpdir, dst)
    check("a later promotion still succeeds",
          open(os.path.join(dst, "new.txt")).read() == "replacement")
finally:
    os.rename, time.sleep = real_rename, real_sleep
    shutil.rmtree(rollback_root, ignore_errors=True)

print("\nALL MATERIALIZE RACE CHECKS PASS")
