#!/usr/bin/env bash
# Run the full benchmark suite: Python side, then R side.
# Extra args pass through to both runners. Use --key=value form so both
# argparse (Python) and the R arg-parser accept the same flags:
#   ./run.sh --reps=5 --scales=1,10 --limit=30
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/.."

echo "==> bench.py"
python benchmarks/bench.py "$@"

echo
echo "==> bench.R"
if command -v Rscript >/dev/null 2>&1; then
  Rscript benchmarks/bench.R "$@"
else
  echo "Rscript not found on PATH; skipping R benchmarks." >&2
fi

echo
echo "Results:"
ls -la benchmarks/results/
