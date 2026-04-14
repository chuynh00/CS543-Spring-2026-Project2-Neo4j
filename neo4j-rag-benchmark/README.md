# Neo4j RAG Benchmark Workspace

This workspace benchmarks two retrieval modes against the same live Neo4j server:

- `native`: Native RAG ON via `CALL rag.retrieve(...)`
- `baseline`: Native RAG OFF via the standard two-call baseline

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
    neo4j_local.example.json
  data/
    query_sets/
      example_manifest.json
  queries/
    cases/
      example-1/
        native.cql
        baseline_vector.cql
        baseline_traversal.cql
  results/
    raw/
    summaries/
    logs/
  scripts/
    bootstrap_python.sh
    prepare_runtime.sh
    run_benchmark.sh
```

## Quick Start

1. Create the Python environment:

```bash
cd neo4j-rag-benchmark
./scripts/bootstrap_python.sh
```

2. Create `config/neo4j_local.json` from the example and point it at your running Neo4j instance.

3. Run one method only:

```bash
./scripts/run_benchmark.sh --method native
./scripts/run_benchmark.sh --method baseline
```

4. Run the full comparison with a restart between phases:

```bash
./scripts/run_benchmark.sh --run-comparison --server-home ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

Comparison mode writes:
- separate CSVs for `native_rag` and `two_call_baseline`
- separate per-method summary JSON files
- one shared log file: `results/logs/comparison-<timestamp>.txt`

5. If you want to reverse the order for a second pass:

```bash
./scripts/run_benchmark.sh --run-comparison --order baseline-first --server-home ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

## Manifest Format

Each benchmark manifest is a JSON array. Each entry provides the three query files for one test case plus the parameter object used to run them.

Example:

```json
[
  {
    "query_id": "example-1",
    "native_query": "cases/example-1/native.cql",
    "baseline_vector_query": "cases/example-1/baseline_vector.cql",
    "baseline_traversal_query": "cases/example-1/baseline_traversal.cql",
    "params": {
      "index_name": "doc_embedding_index",
      "embedding": [1.0, 0.0, 0.0],
      "top_k": 2,
      "depth": 1
    }
  }
]
```
