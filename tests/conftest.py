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

    Used by ``load_dataset`` (above) and by ``test_smooths_predict`` to
    re-attach factor types to ``predict_data.csv`` fixtures, which lose them
    on CSV round-trip just like the source datasets do.
    """
    from lmpy.data import _apply_dataset_schema
    schema_path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    return _apply_dataset_schema(df, schema_path)


def ordered_schema_cols(pkg: str, name: str) -> frozenset[str]:
    """Columns marked `ordered: true` in the dataset's schema sidecar.

    The ``ordered`` flag is plumbed separately from level order — pl.Enum
    carries levels for both ordered and unordered factors, so this is what
    drives `lmpy.formula.with_ordered_cols(...)` for poly contrasts.
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
    """Test-side dataset loader. Delegates to ``lmpy.data.data`` (which
    routes to ``rdatasets`` when covered, bundled CSV otherwise) and caches
    the result so repeated fixture loads are cheap.

    Drops the ``rowname`` column (R's row.names preserved on the bundled-CSV
    side, ``rownames`` injected on the rdatasets side). All R-side fixtures
    were generated without it, so ``y ~ .`` expansions and column lists
    would mismatch otherwise. User-facing ``lmpy.data.data`` keeps the
    column — that's the whole point of preserving meaningful row names
    like the Galápagos island IDs in ``faraway::gala``.
    """
    from lmpy.data import data as _data
    key = (pkg, name)
    if key not in _data_cache:
        df = _data(name, _pkg_subdir(pkg))
        if "rowname" in df.columns:
            df = df.drop("rowname")
        _data_cache[key] = df
    # `lmpy.data` already registers ordered-factor columns globally, but the
    # autouse `_reset_ordered_cols` fixture clears them per-test. Re-register
    # here so the contextvar accumulates across multiple loads inside one test.
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
