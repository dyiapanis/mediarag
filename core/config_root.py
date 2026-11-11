"""Corpus root resolution, shared by the plugin entry and the tool layer.

Kept dependency-free on purpose: the plugin __init__ must be able to compute a
root without importing numpy/onnxruntime/transformers, and the stdio MCP lane
must resolve the same root as the native lane.

Precedence: $MEDIARAG_CONFIG file > plugin/package config file > $PLUGIN_ROOT
config file > $MEDIARAG_ROOT > ~/.mediarag. The last two are only consulted when
no config file yields a root.
"""
import os

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_root(base_dir: str | None = None) -> str:
    """The `root:` value from mediarag-config.yaml, or "" when there is none.

    # ponytail: flat `root:` line parse — no yaml dep on the stdio spawn path;
    # switch to yaml.safe_load if the config ever grows beyond one key.
    """
    candidates = []
    explicit = os.environ.get("MEDIARAG_CONFIG")
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(base_dir or _PKG_DIR, "mediarag-config.yaml"))
    plugin_root = os.environ.get("PLUGIN_ROOT")
    if plugin_root:
        candidates.append(os.path.join(plugin_root, "mediarag-config.yaml"))
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip().startswith("root:"):
                        return line.split(":", 1)[1].strip().strip("'\"")
        except OSError:
            continue
    return ""


def default_root(base_dir: str | None = None) -> str:
    return config_root(base_dir) or os.environ.get("MEDIARAG_ROOT") or os.path.expanduser("~/.mediarag")
