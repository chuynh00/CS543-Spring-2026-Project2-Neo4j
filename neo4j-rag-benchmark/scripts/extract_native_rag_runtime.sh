#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root and the sibling runtime workspace from the benchmark script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_REPO}/.." && pwd)"
PACKAGING_TARGET="${SOURCE_REPO}/packaging/standalone/target"
RUNTIME_ROOT="${PROJECT_ROOT}/neo4j-rag-runtime"

VERSION="${1:-5.26.0}"
FORCE_FLAG="${2:-}"
DIST_NAME="neo4j-community-${VERSION}-native-rag"
ARCHIVE_PATH="${PACKAGING_TARGET}/${DIST_NAME}.tar.gz"
SERVER_HOME="${RUNTIME_ROOT}/${DIST_NAME}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Archive not found: ${ARCHIVE_PATH}"
  echo "Build it first with ./scripts/build_native_rag_tar.sh"
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"

# Avoid wiping an extracted runtime unless the caller opts in.
if [[ -d "${SERVER_HOME}" ]]; then
  if [[ "${FORCE_FLAG}" != "--force" ]]; then
    echo "Runtime directory already exists: ${SERVER_HOME}"
    echo "Re-run with --force to replace it, for example:"
    echo "  ./scripts/extract_native_rag_runtime.sh ${VERSION} --force"
    exit 1
  fi

  rm -rf "${SERVER_HOME}"
fi

# Extract the packaged server into the dedicated runtime workspace.
tar -xzf "${ARCHIVE_PATH}" -C "${RUNTIME_ROOT}"

echo "Extracted runtime to ${SERVER_HOME}"
echo "Next steps:"
echo "  1. cd ${BENCHMARK_ROOT}"
echo "  2. ./scripts/prepare_runtime.sh ${SERVER_HOME}"
echo "  3. ${SERVER_HOME}/bin/neo4j start"
