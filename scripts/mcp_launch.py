#!/usr/bin/env python3
"""MediaRAG MCP launcher — find an interpreter with the engine, then hand over.

The stdio server needs numpy, onnxruntime, transformers and torch (~500 MB).
Agent Plugins v1 gives an MCP entry a `command` token, and `python3` on someone
else's machine is the system python with none of them: the server dies during
the handshake and the client reports only "Connection closed". This launcher is
stdlib-only, so any python3 can start it, and it re-execs the real server under
an interpreter that can actually import the stack.

Resolution order:
  1. $MEDIARAG_PYTHON                    explicit override, always wins
  2. $PLUGIN_DATA/venv/bin/python       per-plugin venv, created once
     ($PLUGIN_DATA/venv/Scripts/python.exe on Windows)
  3. this interpreter, if it already imports the engine
  4. give up, and print the exact command that fixes it

Exit 3 means "no usable interpreter" — a diagnosable message in the client log
instead of a silent disconnect.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "mcp_server.py")
REPO = os.path.dirname(HERE)
REQUIREMENTS = os.path.join(REPO, "requirements.txt")
PROBE = "import numpy, onnxruntime, transformers, torch"
TIMEOUT = 180


def _venv_python(base: str) -> str:
    return (os.path.join(base, "venv", "Scripts", "python.exe") if os.name == "nt"
            else os.path.join(base, "venv", "bin", "python"))


def _usable(interp: str) -> bool:
    if not interp or not os.path.isfile(interp):
        return False
    try:
        return subprocess.run([interp, "-c", PROBE], capture_output=True,
                              timeout=TIMEOUT).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> int:
    data = os.environ.get("PLUGIN_DATA") or os.path.expanduser("~/.mediarag")
    candidates = [os.environ.get("MEDIARAG_PYTHON", ""), _venv_python(data), sys.executable]
    for interp in candidates:
        if not _usable(interp):
            continue
        argv = [interp, SERVER, *sys.argv[1:]]
        if os.name == "nt":                      # no exec semantics we can rely on
            return subprocess.run(argv).returncode
        os.execv(interp, argv)                   # POSIX: replace this process
        return 0                                 # unreachable

    venv = _venv_python(data)
    print(
        "mediarag: no Python interpreter with the engine stack was found.\n"
        "  needs: numpy, onnxruntime, transformers, torch (~500 MB)\n"
        f"  tried: {[c for c in candidates if c]}\n"
        "  fix, once:\n"
        f"    python3 -m venv {os.path.join(data, 'venv')}\n"
        f"    {venv} -m pip install -r {REQUIREMENTS}\n"
        "  or point MEDIARAG_PYTHON at an interpreter that already has them.",
        file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
