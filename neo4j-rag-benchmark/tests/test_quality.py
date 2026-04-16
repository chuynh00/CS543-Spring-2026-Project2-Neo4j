from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark.clients import RetrievalRow
from benchmark.quality import (
    MethodReference,
    build_quality_row,
    build_quality_summary,
    jaccard,
    label_entropy,
    majority_label_fraction,
    mean,
    mean_pairwise_neighborhood_overlap,
    result_jaccard_overlap,
    write_comparison_quality_artifacts,
    write_quality_csv,
)
from benchmark.workloads import QueryCase


class FakeQualityClient:
    def fetch_one_hop_neighborhoods(self, node_ids):
        return {node_id: {node_id + 10} for node_id in node_ids}

    def fetch_node_labels(self, node_ids):
        return {node_id: node_id % 2 for node_id in node_ids}


class QualityMetricsTest(unittest.TestCase):
    def test_jaccard_and_result_overlap(self) -> None:
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertAlmostEqual(result_jaccard_overlap([1, 2], [2, 3]), 1 / 3)
        self.assertIsNone(jaccard(set(), set()))

    def test_mean_pairwise_neighborhood_overlap(self) -> None:
        overlap = mean_pairwise_neighborhood_overlap(
            [1, 2, 3],
            {
                1: {10, 11},
                2: {11, 12},
                3: {30},
            },
        )

        self.assertAlmostEqual(overlap, (1 / 3 + 0 + 0) / 3)
        self.assertIsNone(mean_pairwise_neighborhood_overlap([1], {1: {10}}))

    def test_mean_seed_score(self) -> None:
        reference = MethodReference(
            [
                RetrievalRow(node_id=1, hop_depth=0, score=0.9),
                RetrievalRow(node_id=2, hop_depth=0, score=0.7),
                RetrievalRow(node_id=3, hop_depth=1, score=None),
            ]
        )

        self.assertEqual(reference.seed_ids, [1, 2])
        self.assertAlmostEqual(mean(reference.seed_scores), 0.8)

    def test_label_entropy_and_majority_fraction(self) -> None:
        self.assertAlmostEqual(label_entropy([1, 1, 2, 2]), 1.0)
        self.assertAlmostEqual(majority_label_fraction([1, 1, 1, 2]), 0.75)
        self.assertIsNone(label_entropy([]))
        self.assertIsNone(majority_label_fraction([]))

    def test_build_quality_row_extracts_top_k_depth_and_counts(self) -> None:
        case = QueryCase(
            query_id="q1",
            native_query_path="native.cql",
            baseline_vector_query_path="vector.cql",
            baseline_traversal_query_path="traversal.cql",
            native_query="RETURN 1",
            baseline_vector_query="RETURN 1",
            baseline_traversal_query="RETURN 1",
            params={"top_k": 2, "depth": 1},
            native_config={},
        )
        native = MethodReference(
            [
                RetrievalRow(node_id=1, hop_depth=0, score=0.9),
                RetrievalRow(node_id=2, hop_depth=0, score=0.8),
                RetrievalRow(node_id=3, hop_depth=1, score=None),
            ]
        )
        baseline = MethodReference(
            [
                RetrievalRow(node_id=2, hop_depth=0, score=0.95),
                RetrievalRow(node_id=4, hop_depth=0, score=0.85),
            ]
        )

        row = build_quality_row(
            query_case=case,
            native=native,
            baseline=baseline,
            native_neighborhoods={1: {10, 11}, 2: {11}},
            baseline_neighborhoods={2: {11}, 4: {40}},
            labels_by_node={1: 1, 2: 1, 3: 2, 4: 3},
        )

        self.assertEqual(row["top_k"], 2)
        self.assertEqual(row["depth"], 1)
        self.assertEqual(row["native_row_count"], 3)
        self.assertEqual(row["baseline_row_count"], 2)
        self.assertEqual(row["row_count_difference"], 1)
        self.assertAlmostEqual(row["result_jaccard_overlap"], 1 / 4)
        self.assertEqual(row["shared_result_count"], 1)
        self.assertEqual(row["native_unique_label_count"], 2)
        self.assertEqual(row["baseline_unique_label_count"], 2)

    def test_write_quality_csv_and_summary(self) -> None:
        rows = [
            {
                "query_id": "q1",
                "top_k": 1,
                "depth": 0,
                "native_seed_count": 1,
                "baseline_seed_count": 1,
                "native_mean_seed_ann_score": 0.9,
                "baseline_mean_seed_ann_score": 0.8,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "eval" / "quality.csv"
            write_quality_csv(csv_path, rows)
            summary = build_quality_summary(session_id="test", rows=rows, quality_csv_path=csv_path)

            self.assertTrue(csv_path.exists())
            self.assertEqual(summary["overall"]["query_count"], 1)
            self.assertIn("1", summary["by_top_k"])
            self.assertIn("0", summary["by_depth"])

    def test_write_comparison_quality_artifacts_writes_csv_and_summary(self) -> None:
        case = QueryCase(
            query_id="q1",
            native_query_path="native.cql",
            baseline_vector_query_path="vector.cql",
            baseline_traversal_query_path="traversal.cql",
            native_query="RETURN 1",
            baseline_vector_query="RETURN 1",
            baseline_traversal_query="RETURN 1",
            params={"top_k": 2, "depth": 1},
            native_config={},
        )
        native_rows = {
            "q1": [
                RetrievalRow(node_id=1, hop_depth=0, score=0.9),
                RetrievalRow(node_id=2, hop_depth=1, score=None),
            ]
        }
        baseline_rows = {
            "q1": [
                RetrievalRow(node_id=1, hop_depth=0, score=0.9),
                RetrievalRow(node_id=3, hop_depth=1, score=None),
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            csv_path, summary_path, summary = write_comparison_quality_artifacts(
                query_cases=[case],
                native_rows_by_query=native_rows,
                baseline_rows_by_query=baseline_rows,
                client=FakeQualityClient(),
                results_root=Path(tmp),
                session_id="test",
            )

            self.assertTrue(csv_path.exists())
            self.assertEqual(csv_path.parent.name, "eval")
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary_path.name, "comparison_quality-test.json")
            self.assertEqual(summary["overall"]["query_count"], 1)


if __name__ == "__main__":
    unittest.main()
