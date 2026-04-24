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


def load_dataset(pkg: str, name: str) -> pl.DataFrame:
    key = (pkg, name)
    if key not in _data_cache:
        _data_cache[key] = pl.read_csv(DATA_ROOT / pkg / f"{name}.csv", null_values="NA")
    return _data_cache[key].clone()


def fixture_meta(fx_id: str) -> tuple[dict, dict]:
    fx = FIXTURE_ROOT / fx_id
    return (
        json.loads((fx / "meta.json").read_text()),
        json.loads((fx / "X_meta.json").read_text()),
    )


def fixture_X_ref(fx_id: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURE_ROOT / fx_id / "X.csv", null_values="NA")
