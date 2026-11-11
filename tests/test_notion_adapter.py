"""Notion adapter tests — offline only. No network, no token.

Run: python tests/test_notion_adapter.py
"""
import importlib.machinery
import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "notion_adapter", os.path.join(ROOT, "adapters", "notion"),
    loader=importlib.machinery.SourceFileLoader("notion_adapter", os.path.join(ROOT, "adapters", "notion")))
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)

from core import tools  # noqa: E402


def rt(*texts):
    return [{"plain_text": t} for t in texts]


def blk(typ, text="", **kw):
    body = {"rich_text": rt(text)} if text else {}
    body.update(kw)
    return {"id": "b1", "type": typ, typ: body, "has_children": False}


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS {name}")


# --- block conversion -------------------------------------------------------
lines, _ = n.block_lines(blk("paragraph", "hello world"), 0, "tok")
check("paragraph", lines == ["hello world"], lines)
lines, _ = n.block_lines(blk("heading_1", "Title"), 0, "tok")
check("heading_1", lines == ["# Title"], lines)
lines, _ = n.block_lines(blk("bulleted_list_item", "item"), 0, "tok")
check("bullet", lines == ["- item"], lines)
lines, _ = n.block_lines(blk("quote", "quoted"), 0, "tok")
check("quote", lines == ["> quoted"], lines)
lines, _ = n.block_lines(blk("code", "x=1", language="python"), 0, "tok")
check("code fence", lines == ["```python", "x=1", "```"], lines)
lines, _ = n.block_lines(blk("to_do", "done thing", checked=True), 0, "tok")
check("to_do checked", lines == ["- [x] done thing"], lines)
lines, _ = n.block_lines(blk("to_do", "todo thing", checked=False), 0, "tok")
check("to_do unchecked", lines == ["- [ ] todo thing"], lines)
lines, _ = n.block_lines(blk("divider"), 0, "tok")
check("divider", lines == ["---"], lines)
lines, _ = n.block_lines(blk("callout", "note"), 0, "tok")
check("callout", lines == ["> note"], lines)

row = {"id": "r1", "type": "table_row", "has_children": False,
       "table_row": {"cells": [rt("a"), rt("b"), rt("c")]}}
lines, _ = n.block_lines(row, 0, "tok")
check("table_row", lines == ["a | b | c"], lines)

lines, _ = n.block_lines(blk("image"), 0, "tok")
check("image skipped", lines == [], lines)
lines, _ = n.block_lines(blk("some_future_block", "whatever"), 0, "tok")
check("unknown type ignored", lines == [], lines)

# --- rich text concatenation (annotations are irrelevant) -------------------
b = {"id": "x", "type": "paragraph", "paragraph": {"rich_text": [
    {"plain_text": "one ", "annotations": {"bold": True}},
    {"plain_text": "two", "annotations": {"italic": True}},
]}}
lines, _ = n.block_lines(b, 0, "tok")
check("rich_text joins, annotations dropped", lines == ["one two"], lines)

# --- depth cap --------------------------------------------------------------
check("depth cap", n.page_text("x", "tok", depth=n.MAX_DEPTH + 1) == [])

# --- pagination + 404 fallback (network stubbed) ----------------------------
calls = []


def fake_get(path, token, params=None, payload=None):
    calls.append(path)
    if path.startswith("/databases/db1/query"):
        if payload and payload.get("start_cursor"):
            return {"results": [{"id": "p2", "properties": {"Name": {"type": "title", "title": rt("Second")}}}],
                    "has_more": False, "next_cursor": None}
        return {"results": [{"id": "p1", "properties": {"Name": {"type": "title", "title": rt("First")}}}],
                "has_more": True, "next_cursor": "cur2"}
    if path == "/databases/page404/query":
        raise n.urllib.error.HTTPError("u", 404, "not found", {}, None)
    if path == "/pages/page404":
        return {"id": "page404", "properties": {"title": {"type": "title", "title": rt("Solo")}}}
    if path.startswith("/blocks/"):
        return {"results": [blk("paragraph", "body text")], "has_more": False, "next_cursor": None}
    raise AssertionError(path)


real_get = n.get
n.get = fake_get
try:
    out = []
    n.print = lambda *a, **k: out.append(a[0] if a else "")
    os.environ["NOTION_TOKEN"] = "tok"
    sys.argv = ["notion", "fetch", "db1"]
    rc = n.main()
    items = [json.loads(x) for x in out if x.startswith("{")]
    check("database paginates both pages", [i["title"] for i in items] == ["First", "Second"], items)
    check("uri scheme", items[0]["uri"] == "notion://p1", items[0]["uri"])
    check("body includes title and text", items[0]["text"].startswith("# First\n\nbody text"), items[0]["text"][:60])

    out.clear()
    sys.argv = ["notion", "fetch", "page404"]
    rc = n.main()
    items = [json.loads(x) for x in out if x.startswith("{")]
    check("404 falls back to page", [i["title"] for i in items] == ["Solo"], items)
finally:
    n.get = real_get
    if hasattr(n, "print"):
        del n.print

# --- token missing is a clean error, not a traceback ------------------------
os.environ.pop("NOTION_TOKEN", None)
sys.argv = ["notion", "fetch", "db1"]
check("missing token -> exit 1", n.main() == 1)

# --- registration + env plumbing -------------------------------------------
os.environ["NOTION_TOKEN"] = "tok2"
check("scheme registered", "notion" in tools.list_adapters("/tmp/corpus"))
check("env forwards NOTION_TOKEN", tools._adapter_env("notion").get("NOTION_TOKEN") == "tok2")
check("tokenless scheme leaves env alone", tools._adapter_env("fake").get("NOTION_TOKEN") == "tok2")
os.environ.pop("NOTION_TOKEN", None)
check("missing token is stripped for its own scheme",
      "NOTION_TOKEN" not in tools._adapter_env("notion"))

# --- containers must recurse, and properties must be emitted ---------------
for typ in ("table", "column_list", "column"):
    b = {"id": "c1", "type": typ, "has_children": True, typ: {}}
    _, child = n.block_lines(b, 0, "tok")
    check(f"{typ} recurses into children", child == "c1", child)

page = {"id": "px", "properties": {
    "Name": {"type": "title", "title": rt("Task")},
    "Status": {"type": "status", "status": {"name": "Blocked"}},
    "Owner": {"type": "people", "people": [{"name": "Ada"}]},
    "Tags": {"type": "multi_select", "multi_select": [{"name": "infra"}, {"name": "search"}]},
    "Notes": {"type": "rich_text", "rich_text": rt("quokka migration body")},
    "Due": {"type": "date", "date": {"start": "2026-11-12", "end": None}},
    "Done": {"type": "checkbox", "checkbox": False},
}}
props = n.page_properties(page)
joined = "\n".join(props)
check("status property", "Status: Blocked" in joined, joined)
check("people property", "Owner: Ada" in joined, joined)
check("multi_select property", "Tags: infra, search" in joined, joined)
check("rich_text property", "Notes: quokka migration body" in joined, joined)
check("date property", "Due: 2026-11-12" in joined, joined)
check("checkbox property", "Done: no" in joined, joined)

print("\nALL NOTION ADAPTER CHECKS PASS")
