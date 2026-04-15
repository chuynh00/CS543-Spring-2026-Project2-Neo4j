#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root and the sibling runtime workspace from the benchmark script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_REPO}/.." && pwd)"
PACKAGING_TARGET="${SOURCE_REPO}/packaging/standalone/target"
PLUGIN_TARGET_DIR="${SOURCE_REPO}/community/rag-plugin/target"
TMP_BUILD_ROOT="/tmp/neo4j-native-rag-build"

VERSION="5.26.0"
REBUILD_BASE=0
REBUILD_PLUGIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild-base)
      REBUILD_BASE=1
      ;;
    --rebuild-plugin)
      REBUILD_PLUGIN=1
      ;;
    --help|-h)
      echo "Usage: ./scripts/build_native_rag_tar.sh [VERSION] [--rebuild-base] [--rebuild-plugin]"
      echo ""
      echo "By default, reuse the cached stock Neo4j archive and cached neo4j-rag-plugin JAR if they already exist."
      echo "--rebuild-base forces rebuilding packaging/standalone from source."
      echo "--rebuild-plugin forces rebuilding community/rag-plugin from source."
      exit 0
      ;;
    *)
      VERSION="$1"
      ;;
  esac
  shift
done

BASE_DIST_NAME="neo4j-community-${VERSION}"
BASE_ARCHIVE="${PACKAGING_TARGET}/${BASE_DIST_NAME}-unix.tar.gz"
NATIVE_RAG_DIST_NAME="${BASE_DIST_NAME}-native-rag"
NATIVE_RAG_ARCHIVE="${PACKAGING_TARGET}/${NATIVE_RAG_DIST_NAME}.tar.gz"
PLUGIN_JAR="${PLUGIN_TARGET_DIR}/neo4j-rag-plugin-${VERSION}.jar"
BASE_ARCHIVE_URL="${NEO4J_BASE_ARCHIVE_URL:-https://dist.neo4j.org/${BASE_DIST_NAME}-unix.tar.gz}"

mkdir -p "${PACKAGING_TARGET}"

if [[ "${REBUILD_BASE}" -eq 1 || ! -f "${BASE_ARCHIVE}" ]]; then
  if [[ "${REBUILD_BASE}" -eq 1 ]]; then
    echo "[1/4] Building standalone Neo4j distribution from current source..."
    (
      cd "${SOURCE_REPO}"
      mvn -pl packaging/standalone -am -P '!has-sources' -Dmaven.test.skip=true package
    )
  else
    echo "[1/4] Downloading stock standalone Neo4j distribution..."
    echo "      ${BASE_ARCHIVE_URL}"
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to download ${BASE_ARCHIVE_URL}"
      exit 1
    fi
    curl -fL --retry 3 --retry-all-errors -o "${BASE_ARCHIVE}" "${BASE_ARCHIVE_URL}"
  fi
else
  echo "[1/4] Reusing cached standalone Neo4j distribution..."
  ls -lh "${BASE_ARCHIVE}"
fi

if [[ ! -f "${BASE_ARCHIVE}" ]]; then
  echo "Expected standalone archive was not produced or downloaded: ${BASE_ARCHIVE}"
  exit 1
fi

if [[ "${REBUILD_PLUGIN}" -eq 1 || ! -f "${PLUGIN_JAR}" ]]; then
  echo "[2/4] Building neo4j-rag-plugin jar from current source..."
  (
    cd "${SOURCE_REPO}"
    mvn -pl community/rag-plugin -Dmaven.test.skip=true package
  )
else
  echo "[2/4] Reusing cached neo4j-rag-plugin jar..."
  ls -lh "${PLUGIN_JAR}"
fi

if [[ ! -f "${PLUGIN_JAR}" ]]; then
  echo "Expected plugin jar was not produced: ${PLUGIN_JAR}"
  exit 1
fi

# Expand the stock distro into a temp directory so we can rename it and inject the plugin.
echo "[3/4] Repacking distribution with embedded RAG plugin..."
rm -rf "${TMP_BUILD_ROOT}"
mkdir -p "${TMP_BUILD_ROOT}"
cd "${TMP_BUILD_ROOT}"

tar -xzf "${BASE_ARCHIVE}"
rm -rf "${NATIVE_RAG_DIST_NAME}"
mv "${BASE_DIST_NAME}" "${NATIVE_RAG_DIST_NAME}"
mkdir -p "${NATIVE_RAG_DIST_NAME}/plugins"
cp "${PLUGIN_JAR}" "${NATIVE_RAG_DIST_NAME}/plugins/"

tar -czf "${NATIVE_RAG_ARCHIVE}" "${NATIVE_RAG_DIST_NAME}"

# Verify the embedded plugin is present before telling the user the archive is ready.
echo "[4/4] Verifying packaged plugin..."
if ! tar -tzf "${NATIVE_RAG_ARCHIVE}" | grep -q "plugins/neo4j-rag-plugin-${VERSION}.jar"; then
  echo "Plugin jar was not found inside ${NATIVE_RAG_ARCHIVE}"
  exit 1
fi

ls -lh "${NATIVE_RAG_ARCHIVE}"
echo "Created ${NATIVE_RAG_ARCHIVE}"
