"""
Compare lmpy.formula.materialize_bars() against R/lme4's Z and Lambdat template.

For each lme4 fixture:
  1. Load dataset CSV.
  2. Parse + expand formula → ExpandedFormula with bars.
  3. materialize_bars() → Z, Lambdat (theta-indexed template), theta.
  4. Load fixture Z.mtx, Lambdat.mtx, theta.csv → R's ground truth.
  5. Compare shapes first, then values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmpy.formula import parse, expand, materialize_bars  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DATA_ROOT = ROOT / "datasets"


def _load_data(ds: dict) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / ds["pkg"] / f"{ds['name']}.csv")


def _load_theta(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df.iloc[:, 0].to_numpy(dtype=float)


def main() -> int:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    lme4 = [e for e in manifest["entries"]
            if e.get("kind") == "lme4" and e.get("status") == "ok"]

    total = len(lme4)
    passed = 0
    fails: dict[str, list[str]] = {}

    for entry in lme4:
        fid = entry["id"]
        fx = FIXTURE_ROOT / fid
        meta = json.loads((fx / "meta.json").read_text())
        try:
            data = _load_data(meta["dataset"])
            f = parse(meta["formula"])
            ef = expand(f, data_columns=list(data.columns) if "." in meta["formula"] else None)
            got = materialize_bars(ef, data)
        except Exception as e:
            fails.setdefault(f"RAISED:{type(e).__name__}", []).append(
                f"[{fid}] {meta['formula']!r}: {e}"
            )
            continue

        try:
            Z_ref = np.asarray(mmread(fx / "Z.mtx").todense())
            Lt_ref = np.asarray(mmread(fx / "Lambdat.mtx").todense())
            theta_ref = _load_theta(fx / "theta.csv")
        except Exception as e:
            fails.setdefault("FIXTURE_LOAD", []).append(f"[{fid}]: {e}")
            continue

        if got.Z.shape != Z_ref.shape:
            fails.setdefault("Z_SHAPE", []).append(
                f"[{fid}] {meta['formula']!r}: got Z {got.Z.shape} want {Z_ref.shape}"
            )
            continue
        if got.Lambdat.shape != Lt_ref.shape:
            fails.setdefault("LT_SHAPE", []).append(
                f"[{fid}] {meta['formula']!r}: got Lt {got.Lambdat.shape} want {Lt_ref.shape}"
            )
            continue
        if got.theta.shape != theta_ref.shape:
            fails.setdefault("THETA_SHAPE", []).append(
                f"[{fid}] {meta['formula']!r}: got theta {got.theta.shape} want {theta_ref.shape}"
            )
            continue

        try:
            np.testing.assert_allclose(got.Z, Z_ref, rtol=1e-6, atol=1e-8)
        except AssertionError:
            fails.setdefault("Z_VALS", []).append(f"[{fid}] {meta['formula']!r}")
            continue

        if not np.array_equal(got.Lambdat.astype(int), Lt_ref.astype(int)):
            fails.setdefault("LT_VALS", []).append(f"[{fid}] {meta['formula']!r}")
            continue

        try:
            np.testing.assert_allclose(got.theta, theta_ref, rtol=1e-6, atol=1e-8)
        except AssertionError:
            fails.setdefault("THETA_VALS", []).append(f"[{fid}] {meta['formula']!r}")
            continue

        passed += 1

    print(f"LME4 bars: {passed}/{total}")
    for k, items in sorted(fails.items()):
        print(f"\n=== {k} ({len(items)}) ===")
        for s in items[:10]:
            print(f"  {s}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
