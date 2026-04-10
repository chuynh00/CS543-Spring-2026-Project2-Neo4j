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

VERSION="${1:-5.26.0}"
BASE_DIST_NAME="neo4j-community-${VERSION}"
BASE_ARCHIVE="${PACKAGING_TARGET}/${BASE_DIST_NAME}-unix.tar.gz"
NATIVE_RAG_DIST_NAME="${BASE_DIST_NAME}-native-rag"
NATIVE_RAG_ARCHIVE="${PACKAGING_TARGET}/${NATIVE_RAG_DIST_NAME}.tar.gz"
PLUGIN_JAR="${PLUGIN_TARGET_DIR}/neo4j-rag-plugin-${VERSION}.jar"

# Rebuild the stock standalone package from the current source tree.
echo "[1/4] Building standalone Neo4j distribution from current source..."
(
  cd "${SOURCE_REPO}"
  mvn -pl packaging/standalone -am -P '!has-sources' -Dmaven.test.skip=true package
)

if [[ ! -f "${BASE_ARCHIVE}" ]]; then
  echo "Expected standalone archive was not produced: ${BASE_ARCHIVE}"
  exit 1
fi

# Rebuild the current rag-plugin jar so the packaged server includes your latest code.
echo "[2/4] Building neo4j-rag-plugin jar from current source..."
(
  cd "${SOURCE_REPO}"
  mvn -pl community/rag-plugin -Dmaven.test.skip=true package
)

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
