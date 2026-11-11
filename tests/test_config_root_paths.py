"""Config-root resolution: explicit config, PLUGIN_ROOT fallback, env, default.

Regression for the stdio lane defaulting to ~/.mediarag while the native lane
used the configured corpus (t_c847e73c).

Run: python tests/test_config_root_paths.py   (no model)
"""
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import config_root as cr  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


def write_config(path, root_value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"root: {root_value}\n")


saved = {k: os.environ.get(k) for k in ("MEDIARAG_CONFIG", "PLUGIN_ROOT", "MEDIARAG_ROOT")}
for k in saved:
    os.environ.pop(k, None)

tmp = tempfile.mkdtemp(prefix="mediarag-cfg-")
try:
    # no config anywhere -> env root, then ~/.mediarag
    os.environ["MEDIARAG_ROOT"] = "/env/root"
    check("env root when no config", cr.default_root() == "/env/root", cr.default_root())
    os.environ.pop("MEDIARAG_ROOT", None)
    check("default fallback", cr.default_root().endswith(".mediarag"), cr.default_root())

    # PLUGIN_ROOT supplies the config when the package dir has none
    write_config(os.path.join(tmp, "plugin", "mediarag-config.yaml"), "/plugin/root")
    os.environ["PLUGIN_ROOT"] = os.path.join(tmp, "plugin")
    check("PLUGIN_ROOT config resolves", cr.config_root() == "/plugin/root", cr.config_root())
    check("PLUGIN_ROOT config beats env", cr.default_root() == "/plugin/root", cr.default_root())

    # explicit MEDIARAG_CONFIG wins over PLUGIN_ROOT
    write_config(os.path.join(tmp, "explicit.yaml"), "/explicit/root")
    os.environ["MEDIARAG_CONFIG"] = os.path.join(tmp, "explicit.yaml")
    check("explicit config wins", cr.default_root() == "/explicit/root", cr.default_root())

    # and it beats the env var too
    os.environ["MEDIARAG_ROOT"] = "/env/root"
    check("explicit config beats MEDIARAG_ROOT", cr.default_root() == "/explicit/root",
          cr.default_root())
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print("\nALL CONFIG ROOT PATH CHECKS PASS")
