# Neo4j RAG Benchmark Workspace

This workspace drives benchmarks against a live Neo4j server built from the modified source repo.
It uses the official Neo4j Python driver over Bolt so the same client harness can measure:

- the proposed one-call procedure path: `CALL rag.retrieve(...)`
- the standard two-call baseline:
  1. vector similarity query
  2. traversal query over the returned seed ids

## Layout

```text
neo4j-rag-benchmark/
  benchmark/
    clients.py
    metrics.py
    run_benchmark.py
    workloads.py
  config/
    benchmark_defaults.json
    neo4j_local.json
  data/
    query_sets/
  queries/
    rag_retrieve.cql
    baseline_vector_search.cql
    baseline_traversal.cql
  results/
    raw/
    summaries/
  scripts/
    bootstrap_python.sh
    prepare_runtime.sh
    run_benchmark.sh
```

## Quick Start

1. Create the Python environment:

```bash
cd /Users/chelseahuynh/CSCI543-Project-2-Neo4j/neo4j/neo4j-rag-benchmark
./scripts/bootstrap_python.sh
```

2. Extract a Neo4j standalone distribution under:

```text
/Users/chelseahuynh/CSCI543-Project-2-Neo4j/neo4j-rag-runtime/
```

3. Deploy the plugin jar and runtime config:

```bash
./scripts/prepare_runtime.sh
```

4. Start Neo4j from the extracted server directory.

5. Update `config/neo4j_local.json` with the correct credentials, index name, and query-set path.

6. Run the benchmark:

```bash
./scripts/run_benchmark.sh
```

## Query-Set Schema

Each benchmark query is one JSON object per line in a `.jsonl` file under `data/query_sets/`.

Required fields:
- `query_id`
- `embedding`
- `top_k`
- `depth`

Optional fields:
- `config` for the procedure-specific rerank config map

Example:

```json
{"query_id":"q1","embedding":[0.1,0.2,0.3],"top_k":3,"depth":2,"config":{"oversampleFactor":5}}
```
