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
  2. <plugin>/.venv/bin/python          venv beside the package (created once)
  3. $PLUGIN_DATA/venv/bin/python       venv in the client-provided data dir
  4. this interpreter, if it already imports the engine
  5. give up, and print the exact command that fixes it

The Agent Plugins 1.0 spec is explicit that a plugin bundling an executable must
reach it by a plugin-relative `command`, and that portable plugins must not
depend on PATH or ambient variables for resolution — hence this file is named as
`./scripts/mcp_launch.py` in mcp.json and does its own interpreter discovery.

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
    local = (os.path.join(REPO, ".venv", "Scripts", "python.exe") if os.name == "nt"
             else os.path.join(REPO, ".venv", "bin", "python"))
    candidates = [os.environ.get("MEDIARAG_PYTHON", ""), local,
                  _venv_python(data), sys.executable]
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
        "  or point MEDIARAG_PYTHON at an interpreter that already has them.\n"
        f"  the venv beside the package ({local}) is found automatically.",
        file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
