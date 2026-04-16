from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.clients import RetrievalRow
from benchmark.run_benchmark import BenchmarkRunner, DualLogger, build_comparison_summary, parse_args, run_method_phase
from benchmark.workloads import QueryCase


class FakeClient:
    def run_native_rag(self, *, query_text, params, config):
        return [RetrievalRow(node_id=1, hop_depth=0, score=1.0)]

    def run_two_call_baseline(self, *, vector_query_text, traversal_query_text, params):
        return [RetrievalRow(node_id=1, hop_depth=0, score=1.0)], [1]

    def fetch_one_hop_neighborhoods(self, node_ids):
        return {node_id: {node_id + 10} for node_id in node_ids}

    def fetch_node_labels(self, node_ids):
        return {node_id: node_id % 2 for node_id in node_ids}

    def close(self):
        return None


class BenchmarkRunnerTest(unittest.TestCase):
    def test_parse_args_rejects_invalid_mode_combo(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--method", "native", "--run-comparison"])

    def test_run_method_writes_native_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            logger = DualLogger(results_root / "logs" / "native_rag-test.txt")
            runner = BenchmarkRunner(FakeClient(), defaults={"warmup_runs": 1, "measured_runs": 1, "native_config": {}})
            case = QueryCase(
                query_id="q1",
                native_query_path="cases/q1/native.cql",
                baseline_vector_query_path="cases/q1/baseline_vector.cql",
                baseline_traversal_query_path="cases/q1/baseline_traversal.cql",
                native_query="RETURN 1",
                baseline_vector_query="RETURN 1",
                baseline_traversal_query="RETURN 1",
                params={"depth": 1},
                native_config={},
            )
            try:
                artifacts = runner.run_method(
                    method="native",
                    query_cases=[case],
                    session_id="test",
                    results_root=results_root,
                    logger=logger,
                )
            finally:
                logger.close()

            self.assertTrue(artifacts.raw_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(artifacts.summary["method"], "native_rag")

    def test_build_comparison_summary_marks_matching_rows(self) -> None:
        dummy_summary = {"latency_ms": {"count": 1.0}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            from benchmark.run_benchmark import MethodArtifacts

            native = MethodArtifacts(
                method="native",
                raw_path=tmp_path / "native.csv",
                summary_path=tmp_path / "native.json",
                log_path=tmp_path / "native.txt",
                summary=dummy_summary,
                normalized_results={"q1": [(1, 0)]},
                seed_ids={"q1": []},
                reference_rows={"q1": [RetrievalRow(node_id=1, hop_depth=0, score=1.0)]},
            )
            baseline = MethodArtifacts(
                method="baseline",
                raw_path=tmp_path / "baseline.csv",
                summary_path=tmp_path / "baseline.json",
                log_path=tmp_path / "baseline.txt",
                summary=dummy_summary,
                normalized_results={"q1": [(1, 0)]},
                seed_ids={"q1": [1]},
                reference_rows={"q1": [RetrievalRow(node_id=1, hop_depth=0, score=1.0)]},
            )
            summary = build_comparison_summary(
                native_artifacts=native,
                baseline_artifacts=baseline,
                order="native-first",
                session_id="test",
            )

            self.assertTrue(summary["queries"]["q1"]["result_rows_match"])
            self.assertIsNone(summary["queries"]["q1"]["seed_ids_match"])

    def test_comparison_mode_uses_shared_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            case = QueryCase(
                query_id="q1",
                native_query_path="cases/q1/native.cql",
                baseline_vector_query_path="cases/q1/baseline_vector.cql",
                baseline_traversal_query_path="cases/q1/baseline_traversal.cql",
                native_query="RETURN 1",
                baseline_vector_query="RETURN 1",
                baseline_traversal_query="RETURN 1",
                params={"depth": 1},
                native_config={},
            )
            with patch("benchmark.run_benchmark.BenchmarkClient.connect", return_value=FakeClient()):
                artifacts = run_method_phase(
                    method="native",
                    query_cases=[case],
                    session_id="test",
                    results_root=tmp_path / "results",
                    neo4j_config={"database": "neo4j", "uri": "bolt://localhost:7687", "user": "neo4j", "password": "pw"},
                    defaults={"warmup_runs": 1, "measured_runs": 1, "native_config": {}},
                    comparison_mode=True,
                    shared_log_path=tmp_path / "results" / "logs" / "comparison-test.txt",
                )

            self.assertEqual(artifacts.log_path.name, "comparison-test.txt")
            self.assertTrue(artifacts.log_path.exists())


if __name__ == "__main__":
    unittest.main()
