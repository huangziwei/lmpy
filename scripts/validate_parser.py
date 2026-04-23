"""
Round-trip every corpus formula through lmpy.formula.parse.

Loads corpus/{wr,lme4,mgcv,curated}.yaml, calls parse() on each case's
formula, and reports any ParseErrors with id + formula + message. Exit
code is 0 if all 440 parse cleanly, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmpy.formula import parse, ParseError  # noqa: E402

CORPORA = ("wr", "lme4", "mgcv", "curated")


def main() -> int:
    total = 0
    failed: list[tuple[str, str, str]] = []
    for name in CORPORA:
        path = ROOT / "corpus" / f"{name}.yaml"
        doc = yaml.safe_load(path.read_text())
        for case in doc["cases"]:
            total += 1
            fid, formula = case["id"], case["formula"]
            try:
                parse(formula)
            except ParseError as e:
                failed.append((fid, formula, str(e)))
            except Exception as e:  # tokenizer bugs, index errors, etc.
                failed.append((fid, formula, f"{type(e).__name__}: {e}"))

    ok = total - len(failed)
    print(f"parsed: {ok}/{total}")
    if failed:
        print(f"\nFAILURES ({len(failed)}):")
        for fid, formula, msg in failed[:50]:
            print(f"  [{fid}] {formula!r}")
            print(f"         → {msg}")
        if len(failed) > 50:
            print(f"  ... and {len(failed) - 50} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
