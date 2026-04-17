# NeoFuse - A Kernel-Level Express Lane for GraphRAG

This project combines a **Neo4j 5.26.0** with a **`neo4j-rag-plugin`** that adds a single-call graph RAG procedure, plus a **Python benchmark harness** (`neo4j-rag-benchmark`) that compares that native path to a conventional two-query Cypher workflow on the **ogbn-arxiv** graph.

---

## Features

### Stored procedure - `rag.retrieve` 

The plugin exposes **`CALL rag.retrieve(indexName, embedding, topK, depth, config)`**. 
In one kernel execution it does the following:
1. **Vector index search** - Uses the named vector index and nearest-neighbor seek to obtain an oversampled candidate pool (ANN scores in \([0,1]\)).
2. **Graph-aware seed reranking** - For each candidate, collects a bounded **1-hop neighborhood**, then runs a **greedy reranker**: keep the best ANN hit first, then add seeds that stay high-scoring while penalizing **neighborhood overlap** with seeds already chosen (configurable `overlapPenaltyWeight`, oversampling, caps).
3. **Bounded traversal** - From the final seed set, runs **BFS** to depth `depth` (parallel frontier expansion when `parallelism` is not 1).

Each result row includes the **`Node`**, **`hopDepth`** (0 = seed from vector search, 1+ = graph neighbor), and **`score`** (ANN similarity on seeds; neighbors use the seed score propagation rules implemented in the procedure).

### Reranking and execution knobs - `config` map

Passed as the optional fifth argument; keys are merged with defaults in **`RerankConfig`**:

| Key | Role |
| --- | --- |
| `oversampleFactor` | Multiply `topK` to request a larger ANN pool before reranking (capped by `maxCandidateK`). |
| `overlapPenaltyWeight` | How strongly overlapping 1-hop neighborhoods reduce a candidate’s rerank score. |
| `maxCandidateK` | Upper bound on ANN candidates retrieved. |
| `maxNeighborhoodSize` | Cap on neighbors collected per candidate for overlap computation. |
| `overlapHop` | Must be `1` in this version (1-hop overlap only). |
| `parallelism` | `0` = use available processors; `1` = sequential BFS; `>1` caps worker count. |

### Benchmark-facing modes vs. product defaults

- **Rerank mode** (`benchmark_defaults.json`) uses the full reranker defaults (e.g. `oversampleFactor: 5`, nonzero overlap penalty).
- **Parity mode** (`benchmark_defaults_parity.json`) disables reranking behavior for apples-to-apples comparison with the **two-call baseline** (oversample 1, zero overlap penalty, no neighborhood sampling for reranking), so you can measure agreement and latency without the graph-aware selection layer.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `CS543-Spring-2026-Project2-Neo4j/` | Neo4j source tree; **`community/rag-plugin`** is the added plugin |
| `CS543-Spring-2026-Project2-Neo4j/neo4j-rag-benchmark/` | Dataset, CLI and benchmark scripts (primary entry point: `./scripts/bench.sh`) |
| `../results/` | Benchmark outputs |

### Results folder

It stores **raw** per-query CSVs, **summaries** JSON, **logs**, and **plots** everything produced by a benchmark run so we can analyze latency and get insights on the performance.

---

## Development and running setup

### Prerequisites

- **JDK 17** and **`JAVA_HOME`** set appropriately.
- **Apache Maven** 3.8+ (Neo4j recommends `MAVEN_OPTS="-Xmx2048m"` or similar).
- **Python 3** with `venv` support (for the benchmark environment).
- **bash**, **curl** (used when downloading the stock Neo4j distribution in the packaging script).
- **Git** and sufficient **`ulimit -n`** if you run the full Neo4j test suite (Neo4j docs suggest ≥ 40k open files on Linux).

### Build the Neo4j distribution and plugin

From `CS543-Spring-2026-Project2-Neo4j/`:

- Full build (includes tests):  
  `mvn clean install -T1C`
- JARs only, skip tests:  
  `mvn clean install -DskipTests -T1C`

The **`rag-plugin`** is a Maven module under `community/rag-plugin`; its artifact is packaged into the **native-RAG server tarball** by `neo4j-rag-benchmark/scripts/build_native_rag_tar.sh` (see upstream Neo4j `README.asciidoc` for general build/run notes).

### Benchmark workspace (recommended path)

All commands below are from **`CS543-Spring-2026-Project2-Neo4j/neo4j-rag-benchmark/`**.

1. **Interactive CLI**

   ```bash
   ./scripts/bench.sh
   ```

   Then use `doctor`, `setup`, `run`, `results`, etc. (see `neo4j-rag-benchmark/README.md` for the full command list).

2. **One-shot examples**

   ```bash
   ./scripts/bench.sh doctor
   ./scripts/bench.sh setup
   ./scripts/bench.sh run --tier smoke --mode rerank
   ./scripts/bench.sh run --tier smoke --mode parity
   ./scripts/bench.sh results --recent 5
   ```

3. **Local overrides (optional)**  
   Copy `config/bench_local.example.json` to `config/bench_local.json` and set paths such as `runtime_root`, `java_home`, `uri`, `database`, or `results_root`. That file is gitignored.

The CLI resolves settings from: **CLI flags → `bench_local.json` → `config/bench_profiles.json`**. It does not rewrite tracked config files during runs.

### What `setup` does (high level)

- Creates/refreshes **`.venv`** and installs **`requirements.txt`** (`neo4j` driver, `numpy`, `ogb`, `torch`, CLI libraries).
- Prepares **ogbn-arxiv** CSVs and query manifests (`python -m benchmark.prepare_ogbn_arxiv`).
- Builds or reuses a **Neo4j Community + rag-plugin** tarball, extracts it under a configurable **runtime root** (default profile uses `../../neo4j-rag-runtime`), runs **offline import**, starts the server, and waits for the **vector index** to be online.

---

## Benchmark explanation

### What is being compared

| Method | What runs |
| --- | --- |
| **Native** | `CALL rag.retrieve(...)` with parameters and optional `native_config` from the benchmark defaults / manifest. |
| **Baseline** | **Two Cypher calls**: (1) vector search query to get seed nodes and scores, (2) traversal query with `__DEPTH__` substituted and `seed_ids` passed in — same logical workload as the native path but through the normal Cypher planner twice. |

### Tiers (workload size)

Each tier uses a manifest of sampled paper nodes × **12 scenarios** \((\text{topK} \in \{1,3,5,10\}) \times (\text{depth} \in \{0,1,2\})\):

| Tier | Manifest | Approx. scale |
| --- | --- | --- |
| **smoke** | `data/query_sets/ogbn_arxiv/smoke_manifest.json` | 100 nodes × 12 = **1,200** cases |
| **dev** | `dev_manifest.json` | 500 × 12 = **6,000** cases |
| **full** | `full_manifest.json` | 2,000 × 12 = **24,000** cases |

### Modes

- **`rerank`** — Comparison using **`config/benchmark_defaults.json`** (warmup/measured runs + full rerank `native_config`).
- **`parity`** — Comparison using **`config/benchmark_defaults_parity.json`** (reranking effectively off for parity with baseline).
- **`native-only` / `baseline-only`** — Single-method runs for profiling one side.

### Comparison runs and fairness

For **`rerank`** and **`parity`**, the harness can run **both methods in one session** and **restart Neo4j between phases** (`--run-comparison` with `--server-home`) to reduce cross-method cache bias. Order is configurable (`native-first` vs `baseline-first`).

### Outputs and metrics

Under the chosen **`results_root`** (default `neo4j-rag-benchmark/results/`):

- **`raw/`** — Per-run CSVs (e.g. `native_rag-*.csv`, `two_call_baseline-*.csv`).
- **`summaries/`** — Aggregated JSON (e.g. `comparison-*.json` with latency percentiles per method).
- **`logs/`** — Text logs for the session.

The Python module **`benchmark/quality.py`** can compute **agreement-style metrics** between native and baseline (row counts, Jaccard overlap on result node sets, seed statistics, label entropy, etc.) when you run quality analysis on paired outputs — useful to interpret **`rerank`** vs **`parity`** behavior.

---

## Dataset explanation (ogbn-arxiv)

- **Source**: **Open Graph Benchmark (OGB)** dataset **ogbn-arxiv** — a citation network of arXiv papers with **128-dimensional** node features (used as embeddings in the benchmark), **169,343** nodes and **~1.17M** directed citation edges (as expected by the prep script).
- **Preparation**: `python -m benchmark.prepare_ogbn_arxiv` downloads/caches OGB data, writes Neo4j import CSVs, and generates **tiered query manifests** with deterministic sampling (`QUERY_SEED` in the prep script).
- **Graph model in Neo4j**: Papers are imported with features and labels; a **vector index** (default name **`paper_embedding_idx`** in profiles) indexes the embedding property for ANN queries.
- **Manifests** list, per case: `query_id`, paths to **native** and **baseline** query files under `queries/`, and **`params`** (including embedding, `topK`, `depth`, index name). Optional per-case **`native_config`** merges into defaults.

---
