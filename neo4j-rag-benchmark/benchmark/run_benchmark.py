from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from benchmark.clients import BenchmarkClient, RetrievalRow
from benchmark.metrics import ns_to_ms, summarize_latencies
from benchmark.workloads import QueryCase, load_query_cases


ROOT = Path(__file__).resolve().parent.parent
METHOD_LABELS = {
    "native": "native_rag",
    "baseline": "two_call_baseline",
}


@dataclass(frozen=True)
class MethodArtifacts:
    """Artifact paths and comparison data produced by one benchmark method run."""

    method: str
    raw_path: Path
    summary_path: Path
    log_path: Path
    summary: dict[str, Any]
    normalized_results: dict[str, list[tuple[int, int]]]
    seed_ids: dict[str, list[int]]


class DualLogger:
    """Write benchmark progress to stdout and a per-method log file."""

    def __init__(self, path: Path, *, append: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a" if append else "w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._handle.write(f"{message}\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class BenchmarkRunner:
    """Run warmup and measured benchmark trials for one retrieval method."""

    def __init__(self, client: BenchmarkClient, defaults: dict[str, Any]):
        self.client = client
        self.defaults = defaults

    def run_method(
        self,
        *,
        method: str,
        query_cases: list[QueryCase],
        session_id: str,
        results_root: Path,
        logger: DualLogger,
    ) -> MethodArtifacts:
        label = METHOD_LABELS[method]
        raw_dir = results_root / "raw"
        summary_dir = results_root / "summaries"
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        raw_path = raw_dir / f"{label}-{session_id}.csv"
        summary_path = summary_dir / f"{label}-{session_id}.json"

        warmup_runs = int(self.defaults["warmup_runs"])
        measured_runs = int(self.defaults["measured_runs"])
        total_runs = len(query_cases) * (warmup_runs + measured_runs)

        latencies_ms: list[float] = []
        normalized_results: dict[str, list[tuple[int, int]]] = {}
        seed_ids_by_query: dict[str, list[int]] = {}
        query_summaries: dict[str, Any] = {}

        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "query_file",
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

            completed_runs = 0
            for query_case in query_cases:
                reference_rows, reference_seed_ids = self._reference_outputs(method, query_case)
                normalized_reference_rows = normalize_rows(reference_rows)
                normalized_results[query_case.query_id] = normalized_reference_rows
                seed_ids_by_query[query_case.query_id] = reference_seed_ids
                query_summaries[query_case.query_id] = {
                    "query_file": query_file_for_method(query_case, method),
                    "reference_row_count": len(reference_rows),
                    "reference_seed_ids": reference_seed_ids,
                }

                for phase, count in (("warmup", warmup_runs), ("measured", measured_runs)):
                    for run_index in range(count):
                        completed_runs += 1
                        latency_ms, rows, run_seed_ids = self._timed_run(method, query_case)
                        row_match = normalize_rows(rows) == normalized_reference_rows
                        seed_match: bool | str
                        if method == "baseline":
                            seed_match = run_seed_ids == reference_seed_ids
                        else:
                            seed_match = ""

                        if phase == "measured":
                            latencies_ms.append(latency_ms)

                        logger.log(
                            format_progress_line(
                                current=completed_runs,
                                total=total_runs,
                                query_case=query_case,
                                method=label,
                                phase=phase,
                                run_index=run_index,
                                latency_ms=latency_ms,
                                row_count=len(rows),
                            )
                        )

                        writer.writerow(
                            {
                                "query_id": query_case.query_id,
                                "query_file": query_file_for_method(query_case, method),
                                "phase": phase,
                                "method": label,
                                "run_index": run_index,
                                "latency_ms": f"{latency_ms:.6f}",
                                "row_count": len(rows),
                                "result_node_ids_match": row_match,
                                "seed_ids_match": seed_match,
                            }
                        )

        summary = {
            "method": label,
            "session_id": session_id,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "query_count": len(query_cases),
            "latency_ms": summarize_latencies(latencies_ms),
            "queries": query_summaries,
            "raw_results_path": str(raw_path),
            "log_path": str(logger.path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return MethodArtifacts(
            method=method,
            raw_path=raw_path,
            summary_path=summary_path,
            log_path=logger.path,
            summary=summary,
            normalized_results=normalized_results,
            seed_ids=seed_ids_by_query,
        )

    def _reference_outputs(self, method: str, query_case: QueryCase) -> tuple[list[RetrievalRow], list[int]]:
        if method == "native":
            rows = self.client.run_native_rag(
                query_text=query_case.native_query,
                params=query_case.params,
                config=merged_native_config(self.defaults, query_case),
            )
            return rows, []

        rows, seed_ids = self.client.run_two_call_baseline(
            vector_query_text=query_case.baseline_vector_query,
            traversal_query_text=query_case.baseline_traversal_query,
            params=query_case.params,
        )
        return rows, seed_ids

    def _timed_run(self, method: str, query_case: QueryCase) -> tuple[float, list[RetrievalRow], list[int]]:
        if method == "native":
            latency_ms, rows = timed_call(
                self.client.run_native_rag,
                query_text=query_case.native_query,
                params=query_case.params,
                config=merged_native_config(self.defaults, query_case),
            )
            return latency_ms, rows, []

        latency_ms, baseline_result = timed_call(
            self.client.run_two_call_baseline,
            vector_query_text=query_case.baseline_vector_query,
            traversal_query_text=query_case.baseline_traversal_query,
            params=query_case.params,
        )
        rows, seed_ids = baseline_result
        return latency_ms, rows, seed_ids


def merged_native_config(defaults: dict[str, Any], query_case: QueryCase) -> dict[str, Any]:
    """Merge global native config defaults with case-specific overrides."""

    native_config = dict(defaults.get("native_config", {}))
    native_config.update(query_case.native_config)
    return native_config


def query_file_for_method(query_case: QueryCase, method: str) -> str:
    """Return the most useful query-file label for logging and CSV output."""

    if method == "native":
        return query_case.native_query_path
    return f"{query_case.baseline_vector_query_path} + {query_case.baseline_traversal_query_path}"


def format_progress_line(
    *,
    current: int,
    total: int,
    query_case: QueryCase,
    method: str,
    phase: str,
    run_index: int,
    latency_ms: float,
    row_count: int,
) -> str:
    """Format one informative progress line for the terminal and log files."""

    return (
        f"[{current}/{total}] query_id={query_case.query_id} | phase={phase} | method={method} | "
        f"run={run_index + 1} | latency_ms={latency_ms:.6f} | row_count={row_count} | "
        f"query_file={query_file_for_method(query_case, 'native' if method == 'native_rag' else 'baseline')}"
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


def driver_kwargs_for_config(neo4j_config: dict[str, Any]) -> dict[str, Any]:
    """Return the correct Neo4j driver kwargs for optional auth configs."""

    user = neo4j_config.get("user", "")
    password = neo4j_config.get("password", "")
    if bool(user) != bool(password):
        raise ValueError("Neo4j user and password must either both be set or both be omitted.")
    return {"auth": (user, password)} if user else {}
def restart_server(server_home: Path, neo4j_config: dict[str, Any], logger: DualLogger) -> None:
    """Restart Neo4j between comparison phases and wait for Bolt readiness."""

    neo4j_binary = server_home / "bin" / "neo4j"
    if not neo4j_binary.exists():
        raise FileNotFoundError(f"Neo4j executable not found: {neo4j_binary}")

    logger.log("")
    logger.log("Restarting Neo4j between benchmark phases...")
    result = subprocess.run(
        [str(neo4j_binary), "restart"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        logger.log(result.stdout.strip())
    if result.stderr.strip():
        logger.log(result.stderr.strip())

    wait_for_bolt(neo4j_config=neo4j_config, logger=logger)


def wait_for_bolt(*, neo4j_config: dict[str, Any], logger: DualLogger, timeout_seconds: int = 60) -> None:
    """Wait until the Bolt endpoint accepts connections again."""

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    driver_kwargs = driver_kwargs_for_config(neo4j_config)
    while time.time() < deadline:
        driver = GraphDatabase.driver(neo4j_config["uri"], **driver_kwargs)
        try:
            driver.verify_connectivity()
            logger.log("Neo4j is accepting Bolt connections again.")
            return
        except Exception as exc:  # pragma: no cover - depends on local server timing
            last_error = exc
            time.sleep(1)
        finally:
            driver.close()

    raise TimeoutError(f"Neo4j Bolt endpoint did not become ready within {timeout_seconds}s: {last_error}")

def build_comparison_summary(
    *,
    native_artifacts: MethodArtifacts,
    baseline_artifacts: MethodArtifacts,
    order: str,
    session_id: str,
) -> dict[str, Any]:
    """Build a compact comparison artifact after both phases complete."""

    query_ids = sorted(set(native_artifacts.normalized_results) | set(baseline_artifacts.normalized_results))
    comparisons: dict[str, Any] = {}
    for query_id in query_ids:
        native_rows = native_artifacts.normalized_results.get(query_id, [])
        baseline_rows = baseline_artifacts.normalized_results.get(query_id, [])
        native_seed_ids = native_artifacts.seed_ids.get(query_id, [])
        baseline_seed_ids = baseline_artifacts.seed_ids.get(query_id, [])
        seed_ids_match = None if not native_seed_ids else native_seed_ids == baseline_seed_ids
        comparisons[query_id] = {
            "result_rows_match": native_rows == baseline_rows,
            "seed_ids_match": seed_ids_match,
            "native_row_count": len(native_rows),
            "baseline_row_count": len(baseline_rows),
            "native_seed_count": len(native_seed_ids),
            "baseline_seed_count": len(baseline_seed_ids),
        }

    return {
        "session_id": session_id,
        "order": order,
        "native_rag": {
            "summary_path": str(native_artifacts.summary_path),
            "latency_ms": native_artifacts.summary["latency_ms"],
        },
        "two_call_baseline": {
            "summary_path": str(baseline_artifacts.summary_path),
            "latency_ms": baseline_artifacts.summary["latency_ms"],
        },
        "queries": comparisons,
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--case-manifest",
        default=None,
        help="Path to the benchmark case manifest JSON. Defaults to query_manifest in the Neo4j config.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results"),
        help="Directory root for raw/summaries/logs benchmark output.",
    )
    parser.add_argument(
        "--order",
        default="native-first",
        choices=("native-first", "baseline-first"),
        help="Method order for comparison runs.",
    )
    parser.add_argument(
        "--server-home",
        default=None,
        help="Path to the Neo4j runtime home directory. Required for --run-comparison.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--method",
        choices=("native", "baseline"),
        help="Run only one benchmark method.",
    )
    mode_group.add_argument(
        "--run-comparison",
        action="store_true",
        help="Run native first (or baseline first), restart Neo4j, then run the other method.",
    )
    return parser.parse_args(argv)


def resolve_manifest_path(args: argparse.Namespace, neo4j_config: dict[str, Any]) -> Path:
    """Resolve the benchmark case manifest from CLI args or config."""

    manifest_value = args.case_manifest or neo4j_config.get("query_manifest") or neo4j_config.get("query_set")
    if not manifest_value:
        raise ValueError("No case manifest configured. Set query_manifest in the Neo4j config or pass --case-manifest.")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    return manifest_path


def run_method_phase(
    *,
    method: str,
    query_cases: list[QueryCase],
    session_id: str,
    results_root: Path,
    neo4j_config: dict[str, Any],
    defaults: dict[str, Any],
    comparison_mode: bool,
    shared_log_path: Path | None = None,
) -> MethodArtifacts:
    """Run one method phase and write its dedicated CSV, summary, and log files."""

    label = METHOD_LABELS[method]
    log_path = shared_log_path or (results_root / "logs" / f"{label}-{session_id}.txt")
    logger = DualLogger(log_path, append=comparison_mode and shared_log_path is not None and log_path.exists())
    logger.log("Starting Neo4j query performance A/B test...")
    logger.log("=" * 48)
    logger.log(f"Test time: {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    logger.log(f"Database: {neo4j_config['database']}@{neo4j_config['uri']}")
    logger.log(
        "Testing NATIVE_RAG == ON, TWO_CALL_BASELINE == OFF"
        if method == "native"
        else "Testing NATIVE_RAG == OFF, TWO_CALL_BASELINE == ON"
    )
    if comparison_mode:
        logger.log("Comparison mode enabled.")
    logger.log("")

    client = BenchmarkClient.connect(
        uri=neo4j_config["uri"],
        user=neo4j_config.get("user", ""),
        password=neo4j_config.get("password", ""),
        database=neo4j_config["database"],
    )

    try:
        runner = BenchmarkRunner(client, defaults=defaults)
        artifacts = runner.run_method(
            method=method,
            query_cases=query_cases,
            session_id=session_id,
            results_root=results_root,
            logger=logger,
        )
        logger.log("")
        logger.log(f"Wrote raw results to {artifacts.raw_path}")
        logger.log(f"Wrote summary to {artifacts.summary_path}")
        return artifacts
    finally:
        client.close()
        logger.close()


def main() -> None:
    args = parse_args()
    neo4j_config = load_json(Path(args.neo4j_config))
    defaults = load_json(Path(args.defaults))
    manifest_path = resolve_manifest_path(args, neo4j_config)
    query_cases = load_query_cases(manifest_path, ROOT / "queries")

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = Path(args.output_root)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(parents=True, exist_ok=True)

    if args.run_comparison and not args.server_home:
        raise ValueError("--server-home is required when using --run-comparison")

    if args.run_comparison:
        methods = ["native", "baseline"] if args.order == "native-first" else ["baseline", "native"]
        artifacts_by_method: dict[str, MethodArtifacts] = {}
        comparison_log_path = results_root / "logs" / f"comparison-{session_id}.txt"
        for index, method in enumerate(methods):
            artifacts_by_method[method] = run_method_phase(
                method=method,
                query_cases=query_cases,
                session_id=session_id,
                results_root=results_root,
                neo4j_config=neo4j_config,
                defaults=defaults,
                comparison_mode=True,
                shared_log_path=comparison_log_path,
            )
            if index == 0:
                transition_log = DualLogger(comparison_log_path, append=True)
                try:
                    restart_server(Path(args.server_home), neo4j_config, transition_log)
                finally:
                    transition_log.close()

        comparison_summary = build_comparison_summary(
            native_artifacts=artifacts_by_method["native"],
            baseline_artifacts=artifacts_by_method["baseline"],
            order=args.order,
            session_id=session_id,
        )
        comparison_path = results_root / "summaries" / f"comparison-{session_id}.json"
        comparison_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
        print(f"Wrote comparison summary to {comparison_path}")
        return

    artifacts = run_method_phase(
        method=args.method,
        query_cases=query_cases,
        session_id=session_id,
        results_root=results_root,
        neo4j_config=neo4j_config,
        defaults=defaults,
        comparison_mode=False,
    )
    print(f"Wrote raw results to {artifacts.raw_path}")
    print(f"Wrote summary to {artifacts.summary_path}")
    print(f"Wrote log to {artifacts.log_path}")


if __name__ == "__main__":
    main()
