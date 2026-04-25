import json
from pathlib import Path

import polars as pl
import pytest

from lmpy.formula import set_ordered_cols

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
    """Re-cast factor columns from the sidecar schema into pl.Enum.

    CSV round-trip erases R's factor type — quoted numeric levels come back
    as Int64, character levels as String. Without this step, fs/sz/by=factor
    smooths silently take the non-factor fallthrough path, so the R ground
    truth and lmpy's output agree only on that degraded path.

    Polars 1.40+ makes pl.Categorical process-global (shared string cache
    across DataFrames), which merges sibling columns' level pools and
    reorders levels by first-appearance. pl.Enum keeps its declared levels
    per-column, so we use Enum for every factor here regardless of the
    schema's `ordered` flag. Ordered vs. unordered contrasts are driven by
    `ordered_cols` passed through lmpy's public API (e.g. via `ORDERED_COLS`
    below), not by dtype.
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
        exprs.append(pl.col(col).cast(pl.Utf8).cast(pl.Enum(levels)))
    return df.with_columns(exprs) if exprs else df


def ordered_schema_cols(pkg: str, name: str) -> frozenset[str]:
    """Columns marked `ordered: true` in the dataset's schema sidecar.

    Since pl.Enum carries level-order for both ordered and unordered factors
    (see `_apply_schema`), the ordered flag is plumbed separately via
    `lmpy.formula.with_ordered_cols(...)`.
    """
    path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    if not path.exists():
        return frozenset()
    sch = json.loads(path.read_text())
    return frozenset(
        col for col, spec in sch.get("factors", {}).items() if spec.get("ordered")
    )


_current_ordered_cols: "set[str]" = set()


def load_dataset(pkg: str, name: str) -> pl.DataFrame:
    key = (pkg, name)
    if key not in _data_cache:
        df = pl.read_csv(DATA_ROOT / _pkg_subdir(pkg) / f"{name}.csv", null_values="NA")
        _data_cache[key] = _apply_schema(df, pkg, name)
    # Register any ordered factor columns with lmpy's contextvar so the
    # formula machinery can apply poly contrasts / ordered-by handling.
    # Accumulates across calls within a test (some fixtures touch multiple
    # datasets); `_reset_ordered_cols` autouse fixture clears between tests.
    ordered = ordered_schema_cols(pkg, name)
    if ordered:
        _current_ordered_cols.update(ordered)
        set_ordered_cols(frozenset(_current_ordered_cols))
    return _data_cache[key].clone()


@pytest.fixture(autouse=True)
def _reset_ordered_cols():
    """Clear the ordered-cols contextvar and the accumulator before each test
    so cached-dataset fixtures from an earlier test don't bleed ordered labels
    into an unrelated one."""
    _current_ordered_cols.clear()
    set_ordered_cols(frozenset())
    yield
    _current_ordered_cols.clear()
    set_ordered_cols(frozenset())




def fixture_meta(fx_id: str) -> tuple[dict, dict]:
    fx = FIXTURE_ROOT / fx_id
    return (
        json.loads((fx / "meta.json").read_text()),
        json.loads((fx / "X_meta.json").read_text()),
    )


def fixture_X_ref(fx_id: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURE_ROOT / fx_id / "X.csv", null_values="NA")


# ---------------------------------------------------------------------------
# glm() oracle loader — reads the JSON dumped by tests/scripts/make_glm_oracles.R.
# ---------------------------------------------------------------------------
GLM_ORACLE_ROOT = FIXTURE_ROOT / "glm"


def load_glm_oracle(name: str) -> dict:
    """Load a stats::glm() oracle by id (e.g. 'poisson_log_quine').

    Returns the parsed JSON as a dict; numeric scalars are floats, vectors
    are plain Python lists (test code converts to numpy as needed).
    """
    path = GLM_ORACLE_ROOT / name / "oracle.json"
    if not path.exists():
        raise FileNotFoundError(
            f"glm oracle {name!r} not found at {path}; "
            "regenerate via `Rscript tests/scripts/make_glm_oracles.R`"
        )
    return json.loads(path.read_text())
