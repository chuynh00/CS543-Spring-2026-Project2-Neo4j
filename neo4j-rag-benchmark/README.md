# Neo4j RAG Benchmark Workspace

This workspace benchmarks two retrieval modes against the same local Neo4j runtime:

- `native`: `CALL rag.retrieve(...)`
- `baseline`: vector search plus a separate traversal query

The recommended teammate workflow is now the checked-in interactive CLI.

## Interactive Workflow

From `neo4j-rag-benchmark/`:

```bash
./scripts/bench.sh
```

That opens a `bench>` prompt. Inside it, run:

```text
doctor
setup
run --tier smoke --mode rerank
run --tier smoke --mode parity
run --tier dev --mode rerank
results
results --recent 5
status
stop
exit
```

Useful interactive help:

```text
help
help run
help modes
help tiers
```

If `.venv` is missing or the shell dependencies are stale, `bench.sh` now refreshes the benchmark environment before launching the shell.

## Direct Commands

You can still run the same commands without entering the shell first:

```bash
./scripts/bench.sh doctor
./scripts/bench.sh status
./scripts/bench.sh setup [--fresh] [--reimport] [--rebuild-runtime]
./scripts/bench.sh start
./scripts/bench.sh stop
./scripts/bench.sh results [--recent N] [--file PATH] [--kind comparison|method|all]
./scripts/bench.sh run --tier smoke|dev|full --mode rerank|parity|native-only|baseline-only [--order native-first|baseline-first]
```

What the CLI manages for you:

- bootstrapping `.venv` if it does not exist
- preparing `ogbn-arxiv` dataset artifacts
- downloading or reusing the stock Neo4j server archive, then packaging and extracting the benchmark runtime
- copying the plugin and runtime config overrides
- importing `ogbn-arxiv` when the runtime is missing a matching import
- starting Neo4j before benchmark runs
- selecting the correct tier manifest and benchmark defaults for each named mode

The CLI does not mutate repo-tracked config files during runs. It resolves shared settings from:

1. CLI flags
2. `config/bench_local.json` if present
3. `config/bench_profiles.json`

## Local Overrides

Copy the example file if you need machine-specific overrides:

```bash
cp config/bench_local.example.json config/bench_local.json
```

Typical overrides:

- `runtime_root`
- `runtime_home`
- `java_home`
- `uri`
- `database`
- `results_root`

`config/bench_local.json` is ignored by Git.

## Named Benchmark Modes

- `rerank`: comparison run using `config/benchmark_defaults.json`
- `parity`: comparison run using `config/benchmark_defaults_parity.json`
- `native-only`: native method only
- `baseline-only`: baseline method only

Tier mapping:

- `smoke` -> `data/query_sets/ogbn_arxiv/smoke_manifest.json`
- `dev` -> `data/query_sets/ogbn_arxiv/dev_manifest.json`
- `full` -> `data/query_sets/ogbn_arxiv/full_manifest.json`

## Dataset Artifacts

`python -m benchmark.prepare_ogbn_arxiv` writes:

- `data/processed/ogbn-arxiv/papers.csv`
- `data/processed/ogbn-arxiv/cites.csv`
- `data/processed/ogbn-arxiv/manifest.json`
- `data/query_sets/ogbn_arxiv/smoke_manifest.json`
- `data/query_sets/ogbn_arxiv/dev_manifest.json`
- `data/query_sets/ogbn_arxiv/full_manifest.json`

The current first-class dataset is `ogbn-arxiv`.

## Results

Benchmark outputs are written under `results/` by default:

- `results/raw/`
- `results/summaries/`
- `results/logs/`

The CLI prints the resolved runtime, manifest, mode, and compact latency summary after each run.
You can also inspect the newest artifacts later with `results` or browse recent summaries with `results --recent 5`.

## Low-Level Commands

The old scripts are still available for debugging:

```bash
./scripts/bootstrap_python.sh
./scripts/build_native_rag_tar.sh
./scripts/extract_native_rag_runtime.sh 5.26.0 --force
./scripts/prepare_runtime.sh ../../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
./scripts/import_ogbn_arxiv.sh ../../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag --reset
./scripts/run_benchmark.sh --run-comparison --server-home ../../neo4j-rag-runtime/neo4j-community-5.26.0-native-rag
```

`config/neo4j_local.json` remains supported only for low-level/manual benchmark runs. The new CLI does not require editing it.
