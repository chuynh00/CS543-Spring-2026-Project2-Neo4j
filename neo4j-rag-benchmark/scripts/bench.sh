#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BENCHMARK_ROOT}"

if [[ ! -x "${BENCHMARK_ROOT}/.venv/bin/python" ]]; then
  echo "[bench] Python environment missing. Bootstrapping .venv..."
  "${BENCHMARK_ROOT}/scripts/bootstrap_python.sh"
fi

PYTHON_BIN="${BENCHMARK_ROOT}/.venv/bin/python"

if ! "${PYTHON_BIN}" -c "import rich, prompt_toolkit" >/dev/null 2>&1; then
  echo "[bench] Benchmark shell dependencies missing. Refreshing .venv..."
  "${BENCHMARK_ROOT}/scripts/bootstrap_python.sh"
fi

exec "${PYTHON_BIN}" -m benchmark.cli "$@"
