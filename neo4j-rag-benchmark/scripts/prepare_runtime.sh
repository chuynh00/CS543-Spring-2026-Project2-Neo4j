#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_REPO}/.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/neo4j-rag-runtime"
OVERRIDE_FILE="${BENCHMARK_ROOT}/config/neo4j-rag.conf"

SERVER_HOME="${1:-}"
if [[ -z "${SERVER_HOME}" ]]; then
  SERVER_HOME="$(find "${RUNTIME_ROOT}" -maxdepth 1 -type d -name 'neo4j-community-*' | head -n 1 || true)"
fi

if [[ -z "${SERVER_HOME}" || ! -d "${SERVER_HOME}" ]]; then
  echo "Could not find an extracted Neo4j server under ${RUNTIME_ROOT}."
  echo "Extract a standalone Neo4j distribution there, then re-run this script."
  exit 1
fi

if [[ ! -f "${OVERRIDE_FILE}" ]]; then
  echo "Runtime override file not found: ${OVERRIDE_FILE}"
  exit 1
fi

PLUGIN_TARGET_DIR="${SOURCE_REPO}/community/rag-plugin/target"
PLUGIN_JAR="$(find "${PLUGIN_TARGET_DIR}" -maxdepth 1 -type f -name 'neo4j-rag-plugin-*.jar' ! -name '*-javadoc.jar' ! -name '*-sources.jar' ! -name '*-tests.jar' | head -n 1 || true)"

if [[ -z "${PLUGIN_JAR}" ]]; then
  echo "Plugin JAR not found. Building neo4j-rag-plugin first..."
  (cd "${SOURCE_REPO}" && mvn -pl community/rag-plugin -Dmaven.test.skip=true package)
  PLUGIN_JAR="$(find "${PLUGIN_TARGET_DIR}" -maxdepth 1 -type f -name 'neo4j-rag-plugin-*.jar' ! -name '*-javadoc.jar' ! -name '*-sources.jar' ! -name '*-tests.jar' | head -n 1 || true)"
fi

if [[ -z "${PLUGIN_JAR}" ]]; then
  echo "Failed to build or locate the neo4j-rag-plugin JAR."
  exit 1
fi

mkdir -p "${SERVER_HOME}/plugins"
cp "${PLUGIN_JAR}" "${SERVER_HOME}/plugins/"

if [[ ! -f "${SERVER_HOME}/conf/neo4j.conf" ]]; then
  echo "Neo4j config file not found: ${SERVER_HOME}/conf/neo4j.conf"
  exit 1
fi

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  if ! grep -Fqx "${line}" "${SERVER_HOME}/conf/neo4j.conf"; then
    printf '\n%s\n' "${line}" >> "${SERVER_HOME}/conf/neo4j.conf"
  fi
done < <(grep -v '^#' "${OVERRIDE_FILE}" | sed '/^$/d')

echo "Prepared runtime server at ${SERVER_HOME}"
echo "Copied plugin: ${PLUGIN_JAR}"
echo "Updated config: ${SERVER_HOME}/conf/neo4j.conf"
