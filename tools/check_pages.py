#!/usr/bin/env python3
"""Check every demo page against the payload it ships with.

    python tools/check_pages.py

Two checks, both aimed at the same failure: a page that states something the data no longer
says. It happened once — section 06 of the radio demo displayed transformer scores under a
caption reading "a lexicon score", because the numbers were regenerated and the sentence was
not. Prose does not get re-run when the data does.

1. EVERY FIELD THE PAGE READS EXISTS IN THE PAYLOAD.
   Catches the opposite drift: the page asks for `D.scorer.agreement` and the export stopped
   emitting it, so a number silently renders as "undefined" — or worse, as "NaN" next to three
   correct ones.

2. NO DATA CLAIM IS WRITTEN AS LITERAL TEXT.
   A decimal, a "3 of 10", or a thousands-separated count sitting in static prose is a number
   that cannot be re-derived. It must be interpolated from the payload instead. Structural
   numbers — section labels, a season, a viewport width — are allowed by name below, and the
   allowlist is deliberately short: a growing one means the rule is being worked around.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"

#: Literal numbers permitted in static prose, with the reason each is not a data claim.
ALLOWED = {
    "2024": "the season under study",
    "2023": "the season under study",
    "124M": "the model card's training-set size, a fact about the model not the data",
    "125M": "the model's parameter count",
    "47": "the count of F1 lexicon terms, a property of the code not a result",
    "15": "how many of those terms are neutralised",
    "22": "the number of drivers in a season",
    "10": "out of ten, the rating scale",
    "180": "the character cap on a quoted clip",
    "64": "how many clips are quoted",
    "0.5": "the coin-flip baseline named as a hypothetical, not a measured value",
}

#: JavaScript members that can trail a data path. `D.per_race.filter(...)` reads the field
#: `per_race`; `filter` is not part of it. Stripped from the right, repeatedly, so
#: `D.residual.n.toLocaleString` comes back to `residual.n`.
JS_MEMBERS = {
    "length", "map", "filter", "find", "findIndex", "forEach", "reduce", "some", "every",
    "includes", "indexOf", "join", "slice", "sort", "concat", "flat", "flatMap", "reverse",
    "push", "keys", "values", "entries", "toFixed", "toLocaleString", "toString", "split",
    "replace", "trim", "startsWith", "endsWith", "match", "padStart", "repeat", "at",
}


def strip_js(path: str) -> str:
    parts = path.rstrip(".").split(".")
    while len(parts) > 1 and parts[-1] in JS_MEMBERS:
        parts.pop()
    return ".".join(parts)


CLAIM = re.compile(
    r"(?<![\w.\-])("
    r"[+−-]?\d+\.\d+"                    # a decimal: almost always a statistic
    r"|\d+\s+of\s+\d+"                   # "3 of 10"
    r"|\d{1,3}(?:,\d{3})+"               # 2,791
    r"|ρ\s*[+−-]?\d"                     # an explicit correlation
    r")(?![\w])")


def payload_for(app: Path):
    data = app / "data"
    if not data.is_dir():
        return None, None
    js = sorted(data.glob("*.json"))
    if len(js) != 1:
        return None, None
    return js[0], json.loads(js[0].read_text())


def resolve(payload, path: str) -> bool:
    """Does `D.a.b.c` exist? A list is stepped into at index 0 — the page reads rows the same
    way, so an empty list is as broken as a missing key."""
    cur = payload
    for part in path.split("."):
        if isinstance(cur, list):
            if not cur:
                return False
            cur = cur[0]
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def static_prose(html: str) -> list[tuple[str, str]]:
    body = html.split("<script>", 1)[0]
    out = []
    for m in re.finditer(r'<p class="(note|cap|deck|warn)"[^>]*>(.*?)</p>', body, re.S):
        txt = " ".join(re.sub(r"<[^>]+>", " ", m.group(2)).split())
        out.append((m.group(1), txt))
    return out


def check(app: Path) -> list[str]:
    html = (app / "index.html").read_text()
    problems = []
    path, payload = payload_for(app)

    if payload is not None:
        reads = sorted(set(re.findall(r"\bD\.([A-Za-z_][\w.]*)", html.split("<script>", 1)[-1])))
        for raw in reads:
            r = strip_js(raw)
            if not r or r in JS_MEMBERS:
                continue
            if not resolve(payload, r):
                problems.append(f"page reads D.{r}, not in {path.name}")

    for kind, txt in static_prose(html):
        for m in CLAIM.finditer(txt):
            lit = m.group(1)
            if lit.strip() in ALLOWED:
                continue
            where = txt[max(0, m.start() - 45):m.end() + 45]
            problems.append(f'literal "{lit}" in a <p class="{kind}">: …{where}…')
    return problems


def main() -> int:
    apps = sorted(d for d in APPS.iterdir() if (d / "index.html").exists())
    bad = 0
    for app in apps:
        problems = check(app)
        payload, _ = payload_for(app)
        tag = payload.name if payload else "no payload"
        if problems:
            bad += 1
            print(f"✗ {app.name}  ({tag})")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {app.name}  ({tag})")
    print(f"\n{len(apps) - bad} of {len(apps)} pages clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
