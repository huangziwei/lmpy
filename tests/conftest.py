import json
from pathlib import Path

import polars as pl

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = Path(__file__).parent.parent / "datasets"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def fixtures_by_kind(kind: str):
    m = load_manifest()
    return [
        e for e in m["entries"] if e.get("status") == "ok" and e.get("kind") == kind
    ]


_data_cache: dict[tuple[str, str], pl.DataFrame] = {}


def _pkg_subdir(pkg: str) -> str:
    # datasets/R/ mirrors R's built-in `datasets` package.
    return "R" if pkg == "datasets" else pkg


def _apply_schema(df: pl.DataFrame, pkg: str, name: str) -> pl.DataFrame:
    """Re-cast factor columns from the sidecar schema into the lmpy dtype
    convention (pl.Enum = ordered, pl.Categorical = unordered).

    CSV round-trip erases R's factor type — quoted numeric levels come back
    as Int64, character levels as String. Without this step, fs/sz/by=factor
    smooths silently take the non-factor fallthrough path, so the R ground
    truth and lmpy's output agree only on that degraded path.

    Polars has no native ordered-factor dtype. lmpy treats pl.Enum as R's
    ordered factor (poly contrasts, drop-first in s(…, by=…)) and
    pl.Categorical as R's unordered factor. For unordered cols we cast
    through pl.Enum first so the level order from the schema is preserved
    in the resulting Categorical's `cat.get_categories()`.
    """
    path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    if not path.exists():
        return df
    sch = json.loads(path.read_text())
    factors = sch.get("factors", {})
    if not factors:
        return df
    exprs = []
    for col, spec in factors.items():
        if col not in df.columns:
            continue
        levels = [str(v) for v in spec["levels"]]
        e = pl.col(col).cast(pl.Utf8).cast(pl.Enum(levels))
        if not spec.get("ordered"):
            e = e.cast(pl.Categorical)
        exprs.append(e)
    return df.with_columns(exprs) if exprs else df


def load_dataset(pkg: str, name: str) -> pl.DataFrame:
    key = (pkg, name)
    if key not in _data_cache:
        df = pl.read_csv(DATA_ROOT / _pkg_subdir(pkg) / f"{name}.csv", null_values="NA")
        _data_cache[key] = _apply_schema(df, pkg, name)
    return _data_cache[key].clone()




def fixture_meta(fx_id: str) -> tuple[dict, dict]:
    fx = FIXTURE_ROOT / fx_id
    return (
        json.loads((fx / "meta.json").read_text()),
        json.loads((fx / "X_meta.json").read_text()),
    )


def fixture_X_ref(fx_id: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURE_ROOT / fx_id / "X.csv", null_values="NA")
