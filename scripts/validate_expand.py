"""
Compare lmpy.formula.expand() against R's X_meta.json ground truth.

For each fixture we have: `intercept` (bool) and `term_labels` (list[str]).
We parse the formula, call `expand`, and compare. Cases that contain `.` pull
`data_columns` from the dataset CSV.

Only WR-kind fixtures are checked here — lme4 bars and mgcv smooths go through
later milestones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmpy.formula import parse, expand  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DATA_ROOT = ROOT / "datasets"


def load_data_columns(pkg: str, name: str) -> list[str]:
    return list(pd.read_csv(DATA_ROOT / pkg / f"{name}.csv", nrows=0).columns)


def normalize_labels(tl) -> list[str]:
    if tl is None:
        return []
    if isinstance(tl, str):
        return [tl]
    return list(tl)


def main() -> int:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    wr_entries = [e for e in manifest["entries"] if e.get("kind") == "wr" and e.get("status") == "ok"]

    total = 0
    passed = 0
    fails: list[tuple[str, str, str]] = []

    for entry in wr_entries:
        fid = entry["id"]
        fx = FIXTURE_ROOT / fid
        meta = json.loads((fx / "meta.json").read_text())
        xmeta = json.loads((fx / "X_meta.json").read_text())
        formula_src = meta["formula"]
        want_intercept = xmeta.get("intercept", True)
        want_labels = normalize_labels(xmeta.get("term_labels"))

        total += 1
        try:
            f = parse(formula_src)
            data_cols = None
            if "." in formula_src:
                ds = meta["dataset"]
                data_cols = load_data_columns(ds["pkg"], ds["name"])
            ef = expand(f, data_columns=data_cols)
            got_labels = ef.term_labels
            got_intercept = ef.intercept
        except Exception as e:
            fails.append((fid, formula_src, f"RAISED {type(e).__name__}: {e}"))
            continue

        if got_intercept != want_intercept:
            fails.append((fid, formula_src,
                          f"intercept: got {got_intercept}, want {want_intercept}"))
            continue
        if got_labels != want_labels:
            fails.append((fid, formula_src,
                          f"term_labels:\n    got:  {got_labels}\n    want: {want_labels}"))
            continue
        passed += 1

    print(f"WR expand: {passed}/{total} match R")
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for fid, formula, msg in fails[:40]:
            print(f"  [{fid}] {formula!r}")
            print(f"    {msg}")
        if len(fails) > 40:
            print(f"  ... and {len(fails) - 40} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
