#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BENCHMARK_ROOT}"
if [[ ! -d .venv ]]; then
  echo "Python environment not found. Run ./scripts/bootstrap_python.sh first."
  exit 1
fi

source .venv/bin/activate
python -m benchmark.run_benchmark "$@"
