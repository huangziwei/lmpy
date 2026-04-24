"""Python-side formula benchmark.

Times end-to-end "formula string + dataframe -> design matrix" for:
  - lmpy.formula   (this project)
  - formulaic
  - formulae

The formula suite is drawn from tests/fixtures/manifest.json joined with
corpus/*.yaml, filtered to fixtures with status=ok and a known dataset.

Each library is tried on every fixture; libraries that can't handle a given
formula class are recorded as `status=unsupported` with the error message,
so the reader can see coverage, not just speed.

Also writes `benchmarks/results/suite.json` so the R runner reads the exact
same list of fixtures.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as _md
import json
import sys
import time
import timeit
import traceback
from pathlib import Path

import polars as pl
import yaml

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lmpy.formula import (  # noqa: E402
    expand,
    materialize,
    materialize_bars,
    materialize_smooths,
    parse,
)

FIXTURES = ROOT / "tests" / "fixtures"
DATASETS = ROOT / "datasets"
CORPUS = ROOT / "corpus"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "python.csv"
SUITE_OUT = ROOT / "benchmarks" / "results" / "suite.json"


def _version(pkg: str) -> str:
    try:
        return _md.version(pkg)
    except _md.PackageNotFoundError:
        return "unknown"


VERSIONS = {
    "lmpy": _version("lmpy"),
    "formulaic": _version("formulaic"),
    "formulae": _version("formulae"),
    "polars": _version("polars"),
    "pandas": _version("pandas") if _HAS_PANDAS else "not installed",
    "numpy": _version("numpy"),
}


def load_suite() -> list[dict]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    ok = {e["id"]: e for e in manifest["entries"] if e.get("status") == "ok"}
    formulas: dict[str, str] = {}
    for name in ["curated.yaml", "wr.yaml", "lme4.yaml", "mgcv.yaml"]:
        doc = yaml.safe_load((CORPUS / name).read_text()) or {}
        for case in doc.get("cases") or []:
            formulas[case["id"]] = case["formula"]
    out = []
    for fx_id, e in ok.items():
        if fx_id not in formulas:
            continue
        ds = e.get("dataset")
        if not ds or "/" not in ds:
            continue
        out.append(
            {
                "id": fx_id,
                "kind": e["kind"],
                "formula": formulas[fx_id],
                "dataset": ds,
                "rows": e.get("rows"),
                "cols_X": e.get("cols_X"),
            }
        )
    out.sort(key=lambda r: (r["kind"], r["id"]))
    return out


_pd_cache: dict = {}
_pl_cache: dict[str, pl.DataFrame] = {}


def _split_csv_row(line: str) -> list[tuple[str, bool]]:
    """Parse one CSV line, returning (text, was_quoted) per field."""
    fields: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_quotes = False
    field_quoted = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    buf.append('"'); i += 2; continue
                in_quotes = False
            else:
                buf.append(ch)
        else:
            if ch == '"':
                in_quotes = True; field_quoted = True
            elif ch == ",":
                fields.append(("".join(buf), field_quoted))
                buf.clear(); field_quoted = False
            else:
                buf.append(ch)
        i += 1
    fields.append(("".join(buf), field_quoted))
    return fields


def _detect_quoted_columns(path, max_rows: int = 200) -> set[str]:
    """Return column names whose data values are quoted in the CSV.

    R's write.csv quotes character/factor columns and leaves numerics bare.
    pandas.read_csv and R's read.csv both ignore that signal by default and
    will happily coerce "1","2","3" to int. We recover factor semantics by
    honouring the quoting. Char-level scan (not csv.QUOTE_NONNUMERIC) so
    values like NA don't blow up the reader.
    """
    with open(path) as f:
        header_line = f.readline().rstrip("\r\n")
        if not header_line:
            return set()
        header = [t for t, _ in _split_csv_row(header_line)]
        quoted = [False] * len(header)
        for _ in range(max_rows):
            line = f.readline()
            if not line:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = _split_csv_row(line)
            for j, (_, q) in enumerate(fields):
                if j < len(quoted) and q:
                    quoted[j] = True
            if all(quoted):
                break
    # Drop empty-named columns (R's row.names convention); pandas reads those
    # as `Unnamed: 0` and they never appear in formulas.
    return {h for h, q in zip(header, quoted) if q and h}


def load_df_pd(dataset: str):
    if not _HAS_PANDAS:
        raise RuntimeError(
            "pandas is required to benchmark formulaic/formulae; "
            "install pandas or pass --libs lmpy"
        )
    if dataset not in _pd_cache:
        pkg, name = dataset.split("/", 1)
        path = DATASETS / pkg / f"{name}.csv"
        quoted = _detect_quoted_columns(path)
        dtype = {c: "string" for c in quoted} or None
        df = pd.read_csv(path, dtype=dtype)
        # R's default is stringsAsFactors=TRUE for read-in character cols;
        # mirror that so `s(x0)` on a quoted-int column hits the factor path.
        for c in quoted:
            df[c] = df[c].astype("category")
        _pd_cache[dataset] = df
    return _pd_cache[dataset]


def load_df_pl(dataset: str) -> pl.DataFrame:
    if dataset not in _pl_cache:
        pkg, name = dataset.split("/", 1)
        path = DATASETS / pkg / f"{name}.csv"
        quoted = _detect_quoted_columns(path)
        schema_overrides = {c: pl.String for c in quoted} if quoted else None
        df = pl.read_csv(path, null_values="NA", schema_overrides=schema_overrides)
        if quoted:
            df = df.with_columns([
                pl.col(c).cast(pl.Categorical) for c in quoted if c in df.columns
            ])
        _pl_cache[dataset] = df
    return _pl_cache[dataset]


def scale_pd(df, factor: int):
    if factor <= 1:
        return df
    return pd.concat([df] * factor, ignore_index=True)


def scale_pl(df: pl.DataFrame, factor: int) -> pl.DataFrame:
    if factor <= 1:
        return df
    return pl.concat([df] * factor)


# ---- per-library callables -------------------------------------------------
#
# Each callable takes (formula, df, kind, scope) where scope is "full"
# (parse+materialize+bars/smooths) or "parametric" (just the parametric X,
# matching what the fixture oracle actually captured). We fall back from full
# to parametric when the full pipeline hits data-insufficiency (e.g. a smooth
# that needs more unique covariate rows than the fixture data provides).

def _lmpy(formula_str: str, df: pl.DataFrame, kind: str, scope: str = "full"):
    f = parse(formula_str)
    ex = expand(f, list(df.columns))
    X = materialize(ex, df)
    if scope == "parametric":
        return X
    if kind == "lme4" and ex.bars:
        materialize_bars(ex, df)
    elif kind == "mgcv" and ex.smooths:
        materialize_smooths(ex, df)
    return X


def _formulaic(formula_str: str, df, kind: str, scope: str = "full"):
    from formulaic import Formula
    return Formula(formula_str).get_model_matrix(df)


def _formulae(formula_str: str, df, kind: str, scope: str = "full"):
    from formulae import design_matrices
    return design_matrices(formula_str, df)


# Each lib's native DataFrame flavor: lmpy → polars, formulaic/formulae → pandas.
LIBS: list[tuple[str, callable, str]] = [
    ("lmpy", _lmpy, "pl"),
    ("formulaic", _formulaic, "pd"),
    ("formulae", _formulae, "pd"),
]


_DATA_INSUFFICIENT = (
    "fewer unique", "insufficient unique",
    "doesn't match margin count",  # degenerate tensor dims
    "insufficient data", "too few",
)


def _is_data_insufficient(msg: str) -> bool:
    lower = msg.lower()
    return any(p in lower for p in _DATA_INSUFFICIENT)


# ---- timing ----------------------------------------------------------------

# Per-library "library can't handle this formula class" patterns. Anything
# matching is classified as `unsupported`; anything else is `error` (i.e. may
# be a real bug or a coverage gap worth filing).
_UNSUPPORTED_PATTERNS: dict[str, tuple[str, ...]] = {
    "lmpy": ("not supported", "not implemented", "notimplementederror"),
    # formulaic: bars raise FormulaSyntaxError; smooths surface as
    # FactorEvaluationError ("name 's'/'te' is not defined").
    "formulaic": (
        "not supported", "not implemented", "notimplementederror",
        "formulasyntaxerror", "operator `|`", "operator `||`",
        "factorevaluationerror", "is not defined",
    ),
    # formulae: unknown function/contrast names surface as KeyError; ^ and
    # other operators hit ScanError.
    "formulae": (
        "not supported", "not implemented", "notimplementederror",
        "keyerror:", "scanerror", "unexpected character",
    ),
}


def _classify(lib: str, msg: str) -> str:
    pats = _UNSUPPORTED_PATTERNS.get(lib, ())
    lower = msg.lower()
    return "unsupported" if any(p in lower for p in pats) else "error"


def time_one(fn, *args, reps: int, warmup: int) -> dict:
    # Warmup also surfaces any errors before we start the clock.
    for _ in range(warmup):
        fn(*args)
    t = timeit.repeat(lambda: fn(*args), number=1, repeat=reps)
    t.sort()
    return {
        "min_s": t[0],
        "median_s": t[len(t) // 2],
        "max_s": t[-1],
        "n_reps": reps,
    }


def bench_cell(lib: str, fn, fx: dict, df, reps: int, warmup: int) -> dict:
    row = {
        "library": lib,
        "version": VERSIONS.get(lib, "?"),
        "fixture_id": fx["id"],
        "kind": fx["kind"],
        "formula": fx["formula"],
        "dataset": fx["dataset"],
        "n_rows": len(df),
        "scale": None,  # filled by caller
        "scope": "full",
        "status": "ok",
        "error": "",
        "n_reps": reps,
        "min_s": "",
        "median_s": "",
        "max_s": "",
    }

    def _run(scope: str):
        return time_one(fn, fx["formula"], df, fx["kind"], scope,
                        reps=reps, warmup=warmup)

    try:
        row.update(_run("full"))
        return row
    except NotImplementedError as e:
        row["status"] = "unsupported"
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        # Data-insufficiency for a smooth/tensor: the fixture was built with
        # data that can only support parametric-oracle verification. Retry at
        # parametric scope so we still time the work the fixture oracle covers.
        if _is_data_insufficient(msg):
            try:
                row.update(_run("parametric"))
                row["scope"] = "parametric"
                row["error"] = f"degraded: {msg}"
                return row
            except Exception as e2:
                msg = f"{type(e2).__name__}: {e2}".splitlines()[0][:300]
        row["status"] = _classify(lib, msg)
        row["error"] = msg
        return row


def run(out: Path, limit: int | None, reps: int, warmup: int, scales: list[int],
        kinds: list[str] | None, libs: list[str] | None) -> None:
    suite = load_suite()
    if kinds:
        suite = [s for s in suite if s["kind"] in kinds]
    if limit:
        # Stratified: keep roughly equal counts per kind.
        from collections import defaultdict
        bykind = defaultdict(list)
        for s in suite:
            bykind[s["kind"]].append(s)
        per = max(1, limit // max(1, len(bykind)))
        suite = [s for k in bykind for s in bykind[k][:per]]

    SUITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUITE_OUT.write_text(json.dumps(suite, indent=2))

    selected = [(name, fn, flavor) for name, fn, flavor in LIBS
                if not libs or name in libs]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "library", "version", "fixture_id", "kind", "formula",
                "dataset", "n_rows", "scale", "scope", "status", "error",
                "n_reps", "min_s", "median_s", "max_s",
            ],
        )
        w.writeheader()
        total = len(suite) * len(selected) * len(scales)
        i = 0
        t0 = time.time()
        needs_pd = any(flavor == "pd" for _, _, flavor in selected)
        needs_pl = any(flavor == "pl" for _, _, flavor in selected)
        for fx in suite:
            base_pd = load_df_pd(fx["dataset"]) if needs_pd else None
            base_pl = load_df_pl(fx["dataset"]) if needs_pl else None
            for s in scales:
                df_pd = scale_pd(base_pd, s) if needs_pd else None
                df_pl = scale_pl(base_pl, s) if needs_pl else None
                n_rows = len(df_pl) if df_pl is not None else len(df_pd)
                for name, fn, flavor in selected:
                    df = df_pl if flavor == "pl" else df_pd
                    i += 1
                    row = bench_cell(name, fn, fx, df, reps=reps, warmup=warmup)
                    row["scale"] = s
                    w.writerow(row)
                    fh.flush()
                    elapsed = time.time() - t0
                    tag = row["status"]
                    t_str = f"{row['min_s']:.2e}" if isinstance(row["min_s"], float) else "-"
                    print(
                        f"[{i:4d}/{total}] {name:10s} {fx['kind']:5s} "
                        f"{fx['id']:10s} s={s:<3} n={n_rows:<6} "
                        f"{tag:11s} {t_str}  [{elapsed:5.1f}s]",
                        flush=True,
                    )
    print(f"\nWrote {out}")
    print(f"Wrote suite: {SUITE_OUT}")
    print(f"Versions: {json.dumps(VERSIONS)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap suite size (stratified across kinds)")
    ap.add_argument("--reps", type=int, default=7, help="timeit repeats")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--scales", type=str, default="1",
                    help="comma-separated row multipliers, e.g. 1,10,100")
    ap.add_argument("--kinds", type=str, default=None,
                    help="comma-separated kinds: wr,lme4,mgcv")
    ap.add_argument("--libs", type=str, default=None,
                    help="comma-separated libs: lmpy,formulaic,formulae")
    args = ap.parse_args()
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    kinds = [x.strip() for x in args.kinds.split(",")] if args.kinds else None
    libs = [x.strip() for x in args.libs.split(",")] if args.libs else None
    run(args.out, args.limit, args.reps, args.warmup, scales, kinds, libs)


if __name__ == "__main__":
    main()
