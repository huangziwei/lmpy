"""
Compare lmpy.formula.materialize_smooths() against R/mgcv per-smooth fixtures.

For each mgcv fixture:
  1. Load dataset + smooth_meta.json.
  2. parse + expand → ef.smooths.
  3. materialize_smooths(ef, data) → list[list[SmoothBlock]].
  4. For each smooth i, block k: compare our (X, S_1, …, S_p) to
     fixture's smooth_{i}_{k}_X.mtx / smooth_{i}_{k}_S_{j}.mtx.

Skips fixtures where R reported `smoothCon failed` for any smooth
(since there's no ground truth to compare against).

Unimplemented bs dispatches are tallied by class, not flagged as failures —
the output surfaces where to focus implementation work next.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmpy.formula import parse, expand, materialize_smooths  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DATA_ROOT = ROOT / "datasets"


def _load_data(ds: dict) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / ds["pkg"] / f"{ds['name']}.csv")


def main() -> int:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    mgcv = [e for e in manifest["entries"]
            if e.get("kind") == "mgcv" and e.get("status") == "ok"]

    total = 0
    passed = 0
    skipped_err = 0
    not_impl_by_class: Counter[str] = Counter()
    fails: dict[str, list[str]] = {}

    for entry in mgcv:
        fid = entry["id"]
        fx = FIXTURE_ROOT / fid
        meta = json.loads((fx / "meta.json").read_text())
        sm_meta = json.loads((fx / "smooth_meta.json").read_text())

        # Skip fixtures where R's smoothCon failed on any smooth.
        if any("error" in s for s in sm_meta.get("smooths", [])):
            skipped_err += 1
            continue

        total += 1
        try:
            data = _load_data(meta["dataset"])
            f = parse(meta["formula"])
            data_cols = list(data.columns) if "." in meta["formula"] else None
            ef = expand(f, data_columns=data_cols)
            ours = materialize_smooths(ef, data)
        except NotImplementedError as e:
            # Record by smooth class so we can prioritize what to implement.
            for s in sm_meta.get("smooths", []):
                not_impl_by_class[s["class"]] += 1
                break
            continue
        except Exception as e:
            fails.setdefault(f"RAISED:{type(e).__name__}", []).append(
                f"[{fid}] {meta['formula']!r}: {e}"
            )
            continue

        if len(ours) != len(sm_meta["smooths"]):
            fails.setdefault("N_SMOOTHS", []).append(
                f"[{fid}] {meta['formula']!r}: got {len(ours)} smooths want {len(sm_meta['smooths'])}"
            )
            continue

        ok = True
        for i, (ours_blocks, r_meta) in enumerate(zip(ours, sm_meta["smooths"]), start=1):
            if len(ours_blocks) != r_meta["n_blocks"]:
                fails.setdefault("N_BLOCKS", []).append(
                    f"[{fid}] smooth #{i}: got {len(ours_blocks)} blocks want {r_meta['n_blocks']}"
                )
                ok = False
                break
            for k, blk in enumerate(ours_blocks, start=1):
                try:
                    X_ref = np.asarray(mmread(fx / f"smooth_{i}_{k}_X.mtx").todense(), dtype=float)
                except Exception as e:
                    fails.setdefault("FIXTURE_LOAD", []).append(f"[{fid}] smooth_{i}_{k}_X: {e}")
                    ok = False
                    break
                if blk.X.shape != X_ref.shape:
                    fails.setdefault("X_SHAPE", []).append(
                        f"[{fid}] smooth #{i} block {k}: got X {blk.X.shape} want {X_ref.shape}"
                    )
                    ok = False
                    break
                try:
                    np.testing.assert_allclose(blk.X, X_ref, rtol=1e-6, atol=1e-8)
                except AssertionError:
                    fails.setdefault("X_VALS", []).append(
                        f"[{fid}] smooth #{i} block {k} ({r_meta['class']})"
                    )
                    ok = False
                    break
                # Penalties
                if len(blk.S) != r_meta["n_penalties"]:
                    fails.setdefault("N_PEN", []).append(
                        f"[{fid}] smooth #{i} block {k}: got {len(blk.S)} penalties want {r_meta['n_penalties']}"
                    )
                    ok = False
                    break
                for j, S_got in enumerate(blk.S, start=1):
                    try:
                        S_ref = np.asarray(mmread(fx / f"smooth_{i}_{k}_S_{j}.mtx").todense(), dtype=float)
                    except Exception as e:
                        fails.setdefault("FIXTURE_LOAD", []).append(
                            f"[{fid}] smooth_{i}_{k}_S_{j}: {e}"
                        )
                        ok = False
                        break
                    if S_got.shape != S_ref.shape:
                        fails.setdefault("S_SHAPE", []).append(
                            f"[{fid}] smooth #{i} block {k} S_{j}: got {S_got.shape} want {S_ref.shape}"
                        )
                        ok = False
                        break
                    try:
                        np.testing.assert_allclose(S_got, S_ref, rtol=1e-6, atol=1e-8)
                    except AssertionError:
                        fails.setdefault("S_VALS", []).append(
                            f"[{fid}] smooth #{i} block {k} S_{j} ({r_meta['class']})"
                        )
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            passed += 1

    print(f"MGCV smooths: {passed}/{total}  (skipped smoothCon-failed: {skipped_err})")
    if not_impl_by_class:
        print("\n=== NOT IMPLEMENTED (fixtures blocked by class) ===")
        for cls, n in sorted(not_impl_by_class.items(), key=lambda kv: -kv[1]):
            print(f"  {cls}: {n}")
    for k, items in sorted(fails.items()):
        print(f"\n=== {k} ({len(items)}) ===")
        for s in items[:10]:
            print(f"  {s}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
