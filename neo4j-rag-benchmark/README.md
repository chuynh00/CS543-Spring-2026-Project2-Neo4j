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
    prepare_ogbn_arxiv.py
    run_benchmark.py
    workloads.py
  config/
    benchmark_defaults.json
    benchmark_defaults_parity.json
    neo4j-rag.conf
    neo4j_local.example.json
  data/
    processed/
    query_sets/
      example_manifest.json
  queries/
    baseline_traversal.cql
    baseline_vector_search.cql
    rag_retrieve.cql
    cases/
      example-1/
        native.cql
        baseline_vector.cql
        baseline_traversal.cql
  results/
    logs/
    raw/
    summaries/
  scripts/
    bootstrap_python.sh
    build_native_rag_tar.sh
    extract_native_rag_runtime.sh
    import_ogbn_arxiv.sh
    prepare_runtime.sh
    run_benchmark.sh
```

## Quick Start

1. Create the Python environment:

```bash
cd neo4j-rag-benchmark
./scripts/bootstrap_python.sh
```

2. Prepare the `ogbn-arxiv` dataset artifacts:

```bash
source .venv/bin/activate
python -m benchmark.prepare_ogbn_arxiv
```

3. Build, extract, and prepare the benchmark runtime:

```bash
./scripts/build_native_rag_tar.sh
./scripts/extract_native_rag_runtime.sh 5.26.0 --force
./scripts/prepare_runtime.sh ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

4. Import `ogbn-arxiv` into the runtime and verify the vector index:

```bash
./scripts/import_ogbn_arxiv.sh ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag --reset
```

5. Create `config/neo4j_local.json` from the example and point `query_manifest` at one of the generated manifests, for example:

```json
{
  "uri": "bolt://localhost:7687",
  "user": "",
  "password": "",
  "database": "neo4j",
  "query_manifest": "data/query_sets/ogbn_arxiv/dev_manifest.json"
}
```

The committed runtime override disables auth for the local benchmark runtime, so blank `user` and `password` are valid when you use `config/neo4j-rag.conf`.

6. Run the benchmark:

```bash
./scripts/run_benchmark.sh --run-comparison --server-home ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

Comparison mode writes:
- separate CSVs for `native_rag` and `two_call_baseline`
- separate per-method summary JSON files
- one shared log file: `results/logs/comparison-<timestamp>.txt`

If you want to reverse the order for a second pass:

```bash
./scripts/run_benchmark.sh --run-comparison --order baseline-first --server-home ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

To run just one method:

```bash
./scripts/run_benchmark.sh --method native
./scripts/run_benchmark.sh --method baseline
```

To run the parity configuration instead of the default rerank config:

```bash
./scripts/run_benchmark.sh \
  --defaults config/benchmark_defaults_parity.json \
  --run-comparison \
  --server-home ../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

## `ogbn-arxiv` Dataset Artifacts

`benchmark.prepare_ogbn_arxiv` writes:

- `data/processed/ogbn-arxiv/papers.csv`
- `data/processed/ogbn-arxiv/cites.csv`
- `data/processed/ogbn-arxiv/manifest.json`
- `data/query_sets/ogbn_arxiv/smoke_manifest.json`
- `data/query_sets/ogbn_arxiv/dev_manifest.json`
- `data/query_sets/ogbn_arxiv/full_manifest.json`

Node model:

- label: `Paper`
- properties: `paperId`, `year`, `label`, `split`, `embedding`

Relationship model:

- `(:Paper)-[:CITES]->(:Paper)` in the original OGB citation direction

Workload policy:

- tiers: `smoke=100`, `dev=500`, `full=2000`
- scenarios: `top_k ∈ {1,3,5,10}` and `depth ∈ {0,1,2}`
- query nodes sampled only from `valid` and `test`
- generated manifests point to the shared query templates under `queries/`

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
    },
    "native_config": {
      "oversampleFactor": 5,
      "overlapPenaltyWeight": 0.3,
      "maxCandidateK": 20,
      "maxNeighborhoodSize": 50,
      "overlapHop": 1
    }
  }
]
```
