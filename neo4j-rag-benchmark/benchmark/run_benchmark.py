from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.clients import BenchmarkClient, RetrievalRow
from benchmark.metrics import ns_to_ms, summarize_latencies
from benchmark.workloads import QueryCase, load_query_cases


ROOT = Path(__file__).resolve().parent.parent


class BenchmarkRunner:
    """Run warmup and measured benchmark trials for both retrieval methods."""

    def __init__(self, client: BenchmarkClient, config: dict[str, Any], defaults: dict[str, Any]):
        self.client = client
        self.config = config
        self.defaults = defaults

    def run(self, query_cases: list[QueryCase], output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        summary_path = output_dir.parent / "summaries" / f"summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        native_latencies: list[float] = []
        baseline_latencies: list[float] = []

        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "phase",
                    "method",
                    "run_index",
                    "latency_ms",
                    "row_count",
                    "result_node_ids_match",
                    "seed_ids_match",
                ],
            )
            writer.writeheader()

            for query_case in query_cases:
                self._run_query_case(query_case, writer, native_latencies, baseline_latencies)

        summary = {
            "native_rag": summarize_latencies(native_latencies),
            "two_call_baseline": summarize_latencies(baseline_latencies),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return raw_path, summary_path

    def _run_query_case(
        self,
        query_case: QueryCase,
        writer: csv.DictWriter,
        native_latencies: list[float],
        baseline_latencies: list[float],
    ) -> None:
        warmup_runs = int(self.defaults["warmup_runs"])
        measured_runs = int(self.defaults["measured_runs"])
        native_config = dict(self.defaults.get("native_config", {}))
        native_config.update(query_case.config)

        reference_native_rows = self.client.run_native_rag(
            index_name=self.config["index_name"],
            embedding=query_case.embedding,
            top_k=query_case.top_k,
            depth=query_case.depth,
            config=native_config,
        )
        reference_baseline_rows, reference_seed_ids = self.client.run_two_call_baseline(
            index_name=self.config["index_name"],
            embedding=query_case.embedding,
            top_k=query_case.top_k,
            depth=query_case.depth,
        )

        for phase, count in (("warmup", warmup_runs), ("measured", measured_runs)):
            for run_index in range(count):
                native_latency, native_rows = timed_call(
                    self.client.run_native_rag,
                    index_name=self.config["index_name"],
                    embedding=query_case.embedding,
                    top_k=query_case.top_k,
                    depth=query_case.depth,
                    config=native_config,
                )
                baseline_latency, baseline_result = timed_call(
                    self.client.run_two_call_baseline,
                    index_name=self.config["index_name"],
                    embedding=query_case.embedding,
                    top_k=query_case.top_k,
                    depth=query_case.depth,
                )
                baseline_rows, baseline_seed_ids = baseline_result

                if phase == "measured":
                    native_latencies.append(native_latency)
                    baseline_latencies.append(baseline_latency)

                native_match = normalize_rows(native_rows) == normalize_rows(reference_baseline_rows)
                baseline_match = normalize_rows(baseline_rows) == normalize_rows(reference_native_rows)
                seed_match = baseline_seed_ids == reference_seed_ids

                writer.writerow(
                    {
                        "query_id": query_case.query_id,
                        "phase": phase,
                        "method": "native_rag",
                        "run_index": run_index,
                        "latency_ms": f"{native_latency:.6f}",
                        "row_count": len(native_rows),
                        "result_node_ids_match": native_match,
                        "seed_ids_match": seed_match,
                    }
                )
                writer.writerow(
                    {
                        "query_id": query_case.query_id,
                        "phase": phase,
                        "method": "two_call_baseline",
                        "run_index": run_index,
                        "latency_ms": f"{baseline_latency:.6f}",
                        "row_count": len(baseline_rows),
                        "result_node_ids_match": baseline_match,
                        "seed_ids_match": seed_match,
                    }
                )


def timed_call(func, /, **kwargs):
    """Measure one function call in milliseconds."""

    start = time.perf_counter_ns()
    result = func(**kwargs)
    end = time.perf_counter_ns()
    return ns_to_ms(end - start), result



def normalize_rows(rows: list[RetrievalRow]) -> list[tuple[int, int]]:
    """Normalize retrieval rows for loose equivalence checks across methods."""

    return sorted((row.node_id, row.hop_depth) for row in rows)



def load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON config file into a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark native rag.retrieve against a two-call Cypher baseline.")
    parser.add_argument(
        "--neo4j-config",
        default=str(ROOT / "config" / "neo4j_local.json"),
        help="Path to the Neo4j connection config JSON.",
    )
    parser.add_argument(
        "--defaults",
        default=str(ROOT / "config" / "benchmark_defaults.json"),
        help="Path to the benchmark defaults JSON.",
    )
    parser.add_argument(
        "--query-set",
        default=None,
        help="Path to a JSONL query set. Defaults to the value inside the Neo4j config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "raw"),
        help="Directory for raw benchmark CSV output.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    neo4j_config = load_json(Path(args.neo4j_config))
    defaults = load_json(Path(args.defaults))
    query_set_path = Path(args.query_set or neo4j_config["query_set"])
    if not query_set_path.is_absolute():
        query_set_path = ROOT / query_set_path

    query_cases = load_query_cases(query_set_path)
    client = BenchmarkClient.connect(
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        query_dir=ROOT / "queries",
        database=neo4j_config["database"],
    )

    try:
        runner = BenchmarkRunner(client, config=neo4j_config, defaults=defaults)
        raw_path, summary_path = runner.run(query_cases, Path(args.output_dir))
        print(f"Wrote raw results to {raw_path}")
        print(f"Wrote summary to {summary_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
