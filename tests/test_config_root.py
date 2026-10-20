#!/usr/bin/env python3
"""Corpus-root resolution: named mediarag-config.yaml beats env beats default.

Direct-run: python3 tests/test_config_root.py (exit 0 = all pass).
"""
import os
import sys
import tempfile

import regex  # noqa: F401 — pre-bind venv copies before engine prepends ~/pylibs2
import PIL  # noqa: F401 — pylibs2/PIL is a broken shadow for this venv (no _imaging)

# repo-root parent (= package parent) so `import mediarag` resolves the package __init__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# package dir itself so `from core import …` resolves (test_ingest_units convention)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mediarag  # one package import; _config_root exercised directly below


def main() -> None:
    orig_dir = mediarag._PLUGIN_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            # 1. named config supplies the root
            with open(os.path.join(td, "mediarag-config.yaml"), "w") as fh:
                fh.write("root: /tmp/fleet-corpus\n")
            mediarag._PLUGIN_DIR = td
            assert mediarag._config_root() == "/tmp/fleet-corpus", "config file must win"
            # 2. a config file without a root key contributes nothing
            with open(os.path.join(td, "mediarag-config.yaml"), "w") as fh:
                fh.write("# no root here\n")
            assert mediarag._config_root() == "", "missing root key must fall through"
            # 3. quoted roots and stray spaces parse
            with open(os.path.join(td, "mediarag-config.yaml"), "w") as fh:
                fh.write('root:   "/tmp/quoted root"\n')
            assert mediarag._config_root() == "/tmp/quoted root", "quotes/spaces must strip"
            # 4. no file at all -> empty -> or-chain continues to env/default
            mediarag._PLUGIN_DIR = os.path.join(td, "missing")
            assert mediarag._config_root() == "", "absent config must fall through"
            # 5. resolution sanity: the module always resolved to SOMETHING at import
            assert mediarag.TOOL_ROOT, "TOOL_ROOT must never be empty"
    finally:
        mediarag._PLUGIN_DIR = orig_dir

    # core seam: DEFAULT_ROOT honors the same config > env > default chain (frozen at
    # import — boot semantics; _config_root re-reads dynamically), so stdio and direct
    # core callers resolve the same corpus as the native plugin lane.
    from core import tools as T
    orig_pkg = T._PKG_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "mediarag-config.yaml"), "w") as fh:
                fh.write("root: /tmp/core-corpus\n")
            T._PKG_DIR = td
            assert T._config_root() == "/tmp/core-corpus", "core must read the same config"
            assert T._root("/tmp/explicit") == "/tmp/explicit", "explicit root must always win"
    finally:
        T._PKG_DIR = orig_pkg
    assert isinstance(T.DEFAULT_ROOT, str) and T.DEFAULT_ROOT, "DEFAULT_ROOT must resolve"
    print("test_config_root: OK")


if __name__ == "__main__":
    main()