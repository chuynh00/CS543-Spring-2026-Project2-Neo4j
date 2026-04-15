#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/import_ogbn_arxiv.sh <SERVER_HOME> [--processed-dir <path>] [--database <name>] [--index-name <name>] [--reset]

Offline-import ogbn-arxiv CSVs into an extracted benchmark runtime, create the vector index,
and verify the imported dataset through cypher-shell.
EOF
}

json_field() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path = sys.argv[1]
field_path = sys.argv[2].split(".")
value = json.loads(open(path, encoding="utf-8").read())
for key in field_path:
    if not isinstance(value, dict) or key not in value:
        print("")
        raise SystemExit(0)
    value = value[key]
if isinstance(value, (dict, list)):
    print(json.dumps(value, sort_keys=True))
elif value is None:
    print("")
else:
    print(value)
PY
}

server_running() {
  "${SERVER_HOME}/bin/neo4j" status >/dev/null 2>&1
}

stop_server_if_running() {
  if server_running; then
    echo "Stopping Neo4j before offline import..."
    "${SERVER_HOME}/bin/neo4j" stop >/dev/null
    for _ in $(seq 1 30); do
      if ! server_running; then
        return
      fi
      sleep 1
    done
    echo "Neo4j did not stop cleanly within 30 seconds."
    exit 1
  fi
}

load_connection_settings() {
  URI="${NEO4J_URI:-}"
  USERNAME="${NEO4J_USER:-}"
  PASSWORD="${NEO4J_PASSWORD:-}"

  if [[ -f "${BENCHMARK_ROOT}/config/neo4j_local.json" ]]; then
    [[ -n "${URI}" ]] || URI="$(json_field "${BENCHMARK_ROOT}/config/neo4j_local.json" "uri")"
    [[ -n "${USERNAME}" ]] || USERNAME="$(json_field "${BENCHMARK_ROOT}/config/neo4j_local.json" "user")"
    [[ -n "${PASSWORD}" ]] || PASSWORD="$(json_field "${BENCHMARK_ROOT}/config/neo4j_local.json" "password")"
  fi

  URI="${URI:-bolt://localhost:7687}"
  USERNAME="${USERNAME:-}"
  PASSWORD="${PASSWORD:-}"

  if [[ -n "${USERNAME}" && -z "${PASSWORD}" ]] || [[ -z "${USERNAME}" && -n "${PASSWORD}" ]]; then
    echo "Neo4j credentials must provide both user and password, or neither."
    exit 1
  fi
}

cypher_shell_cmd() {
  local cmd=("${SERVER_HOME}/bin/cypher-shell" "-a" "${URI}" "-d" "${DATABASE}" "--non-interactive" "--format" "plain")
  if [[ -n "${USERNAME}" ]]; then
    cmd+=("-u" "${USERNAME}" "-p" "${PASSWORD}")
  fi
  printf '%s\0' "${cmd[@]}"
}

run_cypher() {
  local query="$1"
  local cmd=()
  while IFS= read -r -d '' arg; do
    cmd+=("${arg}")
  done < <(cypher_shell_cmd)
  cmd+=("${query}")
  "${cmd[@]}"
}

query_scalar() {
  local query="$1"
  run_cypher "${query}" | awk '
    NF == 0 { next }
    $0 == "value" { seen_header = 1; next }
    seen_header {
      gsub(/\r/, "", $0)
      if ($0 ~ /^".*"$/) {
        sub(/^"/, "", $0)
        sub(/"$/, "", $0)
      }
      print
      exit
    }
  '
}

wait_for_bolt() {
  for _ in $(seq 1 60); do
    if run_cypher "RETURN 1 AS value;" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "Neo4j Bolt endpoint did not become ready within 60 seconds."
  exit 1
}

wait_for_index_online() {
  local state=""
  for _ in $(seq 1 60); do
    state="$(query_scalar "SHOW VECTOR INDEXES YIELD name, state WHERE name = '${INDEX_NAME}' RETURN state AS value;")"
    if [[ "${state}" == "ONLINE" ]]; then
      return
    fi
    sleep 1
  done
  echo "Vector index ${INDEX_NAME} did not become ONLINE within 60 seconds. Last observed state: ${state:-<no row>}."
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${BENCHMARK_ROOT}/scripts/java_env_utils.sh"

SERVER_HOME=""
PROCESSED_DIR="${BENCHMARK_ROOT}/data/processed/ogbn-arxiv"
DATABASE="neo4j"
INDEX_NAME="paper_embedding_idx"
RESET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --database)
      DATABASE="$2"
      shift 2
      ;;
    --index-name)
      INDEX_NAME="$2"
      shift 2
      ;;
    --reset)
      RESET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ -z "${SERVER_HOME}" ]]; then
        SERVER_HOME="$1"
      else
        echo "Unexpected argument: $1"
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${SERVER_HOME}" ]]; then
  usage
  exit 1
fi

ensure_java_env
load_connection_settings

if [[ ! -d "${SERVER_HOME}" ]]; then
  echo "Server home not found: ${SERVER_HOME}"
  exit 1
fi

MANIFEST_PATH="${PROCESSED_DIR}/manifest.json"
PAPERS_CSV="${PROCESSED_DIR}/papers.csv"
CITES_CSV="${PROCESSED_DIR}/cites.csv"
MARKER_DIR="${SERVER_HOME}/.rag-benchmark/${DATABASE}"
MARKER_PATH="${MARKER_DIR}/ogbn-arxiv-import.json"
DATABASE_DIR="${SERVER_HOME}/data/databases/${DATABASE}"
TRANSACTIONS_DIR="${SERVER_HOME}/data/transactions/${DATABASE}"
REPORT_FILE="${SERVER_HOME}/logs/import-ogbn-arxiv.report"

for required in "${MANIFEST_PATH}" "${PAPERS_CSV}" "${CITES_CSV}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required import artifact not found: ${required}"
    exit 1
  fi
done

EXPECTED_NODE_COUNT="$(json_field "${MANIFEST_PATH}" "node_count")"
EXPECTED_EDGE_COUNT="$(json_field "${MANIFEST_PATH}" "edge_count")"
DATASET_FINGERPRINT="$(json_field "${MANIFEST_PATH}" "dataset_fingerprint")"

if [[ ! "${INDEX_NAME}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Unsupported vector index name: ${INDEX_NAME}"
  echo "Use a simple Cypher identifier such as paper_embedding_idx."
  exit 1
fi

if [[ -d "${DATABASE_DIR}" && ${RESET} -eq 0 ]]; then
  echo "Database ${DATABASE} already exists under ${DATABASE_DIR}."
  echo "Re-run with --reset to replace it with the current ogbn-arxiv import."
  exit 1
fi

stop_server_if_running

if [[ ${RESET} -eq 1 ]]; then
  rm -rf "${DATABASE_DIR}" "${TRANSACTIONS_DIR}" "${MARKER_DIR}"
fi

mkdir -p "${MARKER_DIR}" "${SERVER_HOME}/logs"

echo "Importing ogbn-arxiv into ${DATABASE}..."
IMPORT_CMD=(
  "${SERVER_HOME}/bin/neo4j-admin"
  database
  import
  full
  "${DATABASE}"
  "--report-file=${REPORT_FILE}"
  "--array-delimiter=;"
  "--nodes=Paper=${PAPERS_CSV}"
  "--relationships=CITES=${CITES_CSV}"
)
if [[ ${RESET} -eq 1 ]]; then
  IMPORT_CMD+=("--overwrite-destination")
fi
"${IMPORT_CMD[@]}"

echo "Starting Neo4j for post-import verification..."
"${SERVER_HOME}/bin/neo4j" start >/dev/null
wait_for_bolt

run_cypher "CREATE VECTOR INDEX ${INDEX_NAME}
FOR (n:Paper) ON (n.embedding)
OPTIONS {
  indexConfig: {
    \`vector.dimensions\`: 128,
    \`vector.similarity_function\`: 'cosine'
  }
};" >/dev/null

wait_for_index_online

NODE_COUNT="$(query_scalar "MATCH (n:Paper) RETURN count(n) AS value;")"
EDGE_COUNT="$(query_scalar "MATCH ()-[r:CITES]->() RETURN count(r) AS value;")"
INDEX_STATE="$(query_scalar "SHOW VECTOR INDEXES YIELD name, state WHERE name = '${INDEX_NAME}' RETURN state AS value;")"

if [[ "${NODE_COUNT}" != "${EXPECTED_NODE_COUNT}" ]]; then
  echo "Unexpected Paper node count: ${NODE_COUNT} != ${EXPECTED_NODE_COUNT}"
  exit 1
fi

if [[ "${EDGE_COUNT}" != "${EXPECTED_EDGE_COUNT}" ]]; then
  echo "Unexpected CITES relationship count: ${EDGE_COUNT} != ${EXPECTED_EDGE_COUNT}"
  exit 1
fi

if [[ "${INDEX_STATE}" != "ONLINE" ]]; then
  echo "Vector index ${INDEX_NAME} is not ONLINE: ${INDEX_STATE}"
  exit 1
fi

cat > "${MARKER_PATH}" <<EOF
{
  "dataset_name": "ogbn-arxiv",
  "database": "${DATABASE}",
  "index_name": "${INDEX_NAME}",
  "dataset_fingerprint": "${DATASET_FINGERPRINT}",
  "manifest_path": "${MANIFEST_PATH}",
  "papers_csv": "${PAPERS_CSV}",
  "cites_csv": "${CITES_CSV}"
}
EOF

echo "Wrote import marker to ${MARKER_PATH}"
echo "Import report saved to ${REPORT_FILE}"
echo "Verified ${NODE_COUNT} Paper nodes, ${EDGE_COUNT} CITES relationships, and ONLINE index ${INDEX_NAME}"
