#!/usr/bin/env python3
"""Plugin engine bench: engine.py vs goldens (cosine >= 0.999 per golden).

Text goldens: re-embed the deterministic input strings. Image/audio goldens:
re-embed the checked-in fixtures (same 10s-wav discipline as export).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mediarag.engine as eng  # noqa: E402

GOLD = os.path.join(os.path.dirname(__file__), "fixtures", "golden.json")

# golden inputs — the strings fed through the prompt templates (text_title_none
# and text_query* are PREFIXED forms; text_plain is verbatim/unprefixed)
TEXT_INPUTS = {
    "text_title_none": ("doc", "The hippocampus stores the memory, filing it as a pattern for later."),
    "text_query": ("query", "where is the C6 landing clip"),
    "text_query2": ("query", "let us move the meeting to three o'clock on thursday"),
    "text_plain": ("plain", "a whole human brain glowing blue on a dark background"),
}


def cos(a, b) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    gold = json.load(open(GOLD))
    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    fails = []
    for g in gold["goldens"]:
        if g["tower"] == "text":
            lane, raw = TEXT_INPUTS[g["key"]]
            v = {"doc": eng.embed_doc, "query": eng.embed_query, "plain": eng.embed_plain}[lane](raw)
        elif g["tower"] == "vision":
            v = eng.embed_image(os.path.join(fixtures, g["fixture"]))
        elif g["tower"] == "audio":
            v = eng.embed_audio(os.path.join(fixtures, g["fixture"]))
        else:
            continue
        c = cos(v, g["vec"])
        mark = "OK " if c >= 0.999 else "FAIL"
        print(f"{mark} {g['tower']:<7}{g['key']:<20} cos={c:.6f}")
        if c < 0.999:
            fails.append(g["key"])
    if fails:
        print(f"bench FAILED: {fails}")
        return 1
    print("bench: all goldens >= 0.999")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())