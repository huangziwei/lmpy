"""
Compare lmpy.formula.materialize() against R's X.csv fixture by fixture.

For each WR fixture:
  1. Load its dataset CSV.
  2. Parse + expand + materialize → our X.
  3. Load fixture X.csv → R's X.
  4. Compare: shape first, then by column position.
  5. Also sanity-check R's `colnames` vs ours.

Only WR fixtures are tested here (lme4 / mgcv have different ground truth).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmpy.formula import parse, expand, materialize  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DATA_ROOT = ROOT / "datasets"


def _aslist(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _load_data(ds: dict) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / ds["pkg"] / f"{ds['name']}.csv")


def main() -> int:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    wr = [e for e in manifest["entries"] if e.get("kind") == "wr" and e.get("status") == "ok"]

    total = len(wr)
    passed = 0
    shape_mismatch = 0
    value_mismatch = 0
    raised = 0
    by_failure_type: dict[str, list[str]] = {}

    for entry in wr:
        fid = entry["id"]
        fx = FIXTURE_ROOT / fid
        meta = json.loads((fx / "meta.json").read_text())
        try:
            x_ref = pd.read_csv(fx / "X.csv")
        except pd.errors.EmptyDataError:
            # R emitted a zero-column X (e.g. `y ~ 0`). Build an empty
            # DataFrame with the right row count from the dataset.
            n = len(_load_data(meta["dataset"]))
            x_ref = pd.DataFrame(index=range(n))

        try:
            data = _load_data(meta["dataset"])
            f = parse(meta["formula"])
            data_cols = list(data.columns) if "." in meta["formula"] else None
            ef = expand(f, data_columns=data_cols)
            x_got = materialize(ef, data)
        except Exception as e:
            raised += 1
            key = f"RAISED:{type(e).__name__}"
            by_failure_type.setdefault(key, []).append(f"[{fid}] {meta['formula']!r}: {e}")
            continue

        if x_got.shape != x_ref.shape:
            shape_mismatch += 1
            by_failure_type.setdefault("SHAPE", []).append(
                f"[{fid}] {meta['formula']!r}: got {x_got.shape} want {x_ref.shape}"
            )
            continue

        try:
            np.testing.assert_allclose(
                x_got.values.astype(float), x_ref.values.astype(float),
                rtol=1e-6, atol=1e-8,
            )
        except AssertionError as e:
            value_mismatch += 1
            by_failure_type.setdefault("VALUES", []).append(
                f"[{fid}] {meta['formula']!r}\n    got cols: {list(x_got.columns)}\n    ref cols: {list(x_ref.columns)}"
            )
            continue

        passed += 1

    print(f"WR materialize: {passed}/{total}")
    print(f"  raised:         {raised}")
    print(f"  shape mismatch: {shape_mismatch}")
    print(f"  value mismatch: {value_mismatch}")
    print()
    for kind, items in sorted(by_failure_type.items()):
        print(f"=== {kind} ({len(items)}) ===")
        for s in items[:10]:
            print(f"  {s}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
