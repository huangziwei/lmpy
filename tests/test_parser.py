"""
Round-trip every corpus formula through lmpy.formula.parse.

One test per (corpus, case id). Failures show which formulas still break
the parser — there are no xfail markers, a failing test is the signal
that something is unimplemented or has regressed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmpy.formula import ParseError, parse

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
CORPORA = ("wr", "lme4", "mgcv", "curated")


def _load_cases():
    cases = []
    for name in CORPORA:
        doc = yaml.safe_load((CORPUS_DIR / f"{name}.yaml").read_text())
        for case in doc["cases"]:
            cases.append((name, case["id"], case["formula"]))
    return cases


CASES = _load_cases()
IDS = [f"{corpus}/{fid}" for corpus, fid, _ in CASES]


@pytest.mark.parametrize("corpus,fid,formula", CASES, ids=IDS)
def test_parse(corpus: str, fid: str, formula: str):
    try:
        parse(formula)
    except ParseError as e:
        pytest.fail(f"ParseError: {e}")
