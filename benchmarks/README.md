# benchmarks

Fair-as-we-can-make-it speed comparison of `lmpy.formula` against its Python
peers and the R toolchain it's modelled on.

## What's measured

End-to-end cost of the operation:

> formula string + raw dataframe → design matrix (+ random-effect / smooth
> structure, where applicable)

Model fitting (`lm`, `lmer`, `gam`) is deliberately **not** timed — it would
conflate numerics with parsing, and only a thin sliver of that cost is the
formula layer. The bench hits the layer we actually wrote.

## Libraries

| library     | entry point                                | handles         |
|-------------|--------------------------------------------|-----------------|
| `lmpy`      | `parse` → `expand` → `materialize` (+ `_bars`, `_smooths`) | wr + lme4 + mgcv |
| `formulaic` | `Formula(f).get_model_matrix(df)`          | wr              |
| `formulae`  | `design_matrices(f, df)`                   | wr + lme4       |
| `base` (R)  | `model.matrix(terms(f), data)`             | wr (+ mgcv parametric baseline) |
| `lme4` (R)  | `lme4::lFormula(f, data)`                  | lme4            |
| `mgcv` (R)  | `mgcv::gam(f, data, fit = FALSE)`          | mgcv            |

Libraries outside their scope are recorded as `status=unsupported`, not
silently skipped — so a fast number that comes from ignoring half the
grammar shows up as coverage, not speed.

## Suite

Pulled from `tests/fixtures/manifest.json` joined with `corpus/*.yaml`. Every
fixture with `status=ok` and a resolvable dataset under `datasets/<pkg>/` is
included (~300 formulas across `wr`, `lme4`, `mgcv` kinds).

Each fixture ships with real data from its source package (iris, sleepstudy,
Machines, etc.), matching the formulas to data shapes they were authored for.

## Method

- **No rpy2.** Python and R time themselves natively; rpy2's cross-language
  tax would pollute the R numbers. Python writes `results/suite.json`; R
  reads the same file so both runners cover the same fixtures in the same
  order.
- **Repetitions.** Each (library, fixture, scale) cell runs `--warmup` warmup
  iterations and then `--reps` timed iterations. Output is min / median / max
  seconds. Report **min** for CPU-bound microbenchmarks (most noise is
  additive), **median** for larger-scale runs.
- **Scale axis.** `--scales=1,10,100` replicates the dataset rows to see how
  cost splits between fixed parse overhead and per-row materialization.
- **Fair exclusions.** A library that can't handle a formula class gets an
  `unsupported` row with the error/reason, not a missing cell.

## Run it

```bash
# Full suite (several minutes, depending on scales)
./benchmarks/run.sh

# Fast smoke run
./benchmarks/run.sh --limit=30 --reps=3

# Only the scaling story on wr formulas
./benchmarks/run.sh --kinds=wr --scales=1,10,100 --reps=5

# Python only
python benchmarks/bench.py --reps=5 --limit=30

# R only (requires suite.json from a previous bench.py run)
Rscript benchmarks/bench.R --reps=5
```

Outputs:

```
benchmarks/results/
  suite.json      shared fixture list (Python writes, R reads)
  python.csv      lmpy / formulaic / formulae
  r.csv           base / lme4 / mgcv
```

Both CSVs share the same schema — concatenate and group by `library` for
analysis:

```
library, version, fixture_id, kind, formula, dataset,
n_rows, scale, status, error, n_reps, min_s, median_s, max_s
```

## Caveats

- `formulaic` has a materializer cache keyed on the formula object; we
  construct a fresh `Formula(...)` per iteration to measure the cold path,
  since that's what users pay when calling with a string.
- `mgcv::gam(..., fit = FALSE)` still runs the smooth-basis construction and
  penalty assembly; it does not just parse. That's the right comparison for
  `materialize_smooths`, but remember the R side is doing more work than a
  pure parse would.
- Base R's `model.matrix` skips the `na.action` step if the data has no NAs;
  same for formulaic/formulae. Datasets in the suite are mostly NA-free.
- Dataset column types differ slightly between pandas (`read_csv`) and R
  (`read.csv(stringsAsFactors=TRUE)`) — e.g., pandas keeps `object`; R auto-
  converts to `factor`. This is how each ecosystem's users actually call
  these libraries, so we measure them in-idiom rather than forcing parity.
