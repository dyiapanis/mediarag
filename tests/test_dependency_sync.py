#!/usr/bin/env python3
"""The dependency pin is authored three times (requirements.txt, plugin.yaml,
pyproject.toml). PM reads pyproject first, the plugin loader reads the same
declaration, humans read requirements.txt — when one copy drifts the result is a
startup warning that the package is "not installed" while it sits in site-packages.

Direct-run: python3 tests/test_dependency_sync.py (exit 0 = all pass).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
SPEC = re.compile(r"^\s*[A-Za-z0-9_.\-]+\s*(.*)$")


def norm(req: str) -> tuple[str, str]:
    """(normalised name, specifier) for a requirement line."""
    req = req.strip().strip('"').strip("'").strip(",").strip()
    m = NAME.match(req)
    if not m:
        return ("", "")
    name = m.group(1).lower().replace("_", "-")
    rest = SPEC.match(req)
    return (name, re.sub(r"\s+", "", rest.group(1) if rest else ""))


def from_requirements() -> dict[str, str]:
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0]
        if line.strip():
            out[norm(line)[0]] = norm(line)[1]
    return out


def from_plugin_yaml() -> dict[str, str]:
    out, inside = {}, False
    for line in (ROOT / "plugin.yaml").read_text().splitlines():
        if line.startswith("python_dependencies:"):
            inside = True
            continue
        if inside:
            if line.strip().startswith("- "):
                key, spec = norm(line.strip()[2:])
                out[key] = spec
            elif line.strip() and not line.startswith((" ", "\t")):
                inside = False
    return out


def from_pyproject() -> dict[str, str]:
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
    assert block, "pyproject.toml: no [project] dependencies list"
    out = {}
    for raw in re.findall(r'"([^"]+)"', block.group(1)):
        key, spec = norm(raw)
        out[key] = spec
    return out


def main() -> int:
    sources = {
        "requirements.txt": from_requirements(),
        "plugin.yaml": from_plugin_yaml(),
        "pyproject.toml": from_pyproject(),
    }
    for label, deps in sources.items():
        assert deps, f"{label}: parsed zero dependencies"
        print(f"PASS {label}: {len(deps)} dependencies")

    reference = sources["requirements.txt"]
    for label, deps in sources.items():
        assert set(deps) == set(reference), (
            f"{label}: name set differs from requirements.txt "
            f"(+{sorted(set(deps) - set(reference))} -{sorted(set(reference) - set(deps))})")
        for name, spec in deps.items():
            assert spec == reference[name], (
                f"{label}: {name} pin drifted: {spec!r} != {reference[name]!r}")
        print(f"PASS {label} agrees with requirements.txt")

    assert "pillow" in reference, "pillow must stay declared (image ingest needs it)"
    assert reference["pillow"].endswith("<14"), (
        f"pillow pin {reference['pillow']!r} excludes the installed 13.x line")
    print(f"PASS pillow pin {reference['pillow']} admits 13.x")
    print(f"{len(reference)} dependencies in sync across 3 files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
