from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.prepare_ogbn_arxiv import (
    EMBEDDING_DIMENSION,
    EXPECTED_EDGE_COUNT,
    EXPECTED_NODE_COUNT,
    SCENARIOS,
    TIER_SIZES,
    build_split_assignments,
    sample_query_nodes,
    validate_dataset_shape,
    write_query_manifests,
)
from benchmark.workloads import load_query_cases


class PrepareOgbnArxivTest(unittest.TestCase):
    def test_validate_dataset_shape_rejects_wrong_dimension(self) -> None:
        edge_index = SimpleNamespace(shape=(2, EXPECTED_EDGE_COUNT))
        node_features = SimpleNamespace(shape=(EXPECTED_NODE_COUNT, EMBEDDING_DIMENSION - 1))
        node_year = [2000] * EXPECTED_NODE_COUNT
        labels = [0] * EXPECTED_NODE_COUNT

        with self.assertRaisesRegex(ValueError, "embedding dimension"):
            validate_dataset_shape(EXPECTED_NODE_COUNT, edge_index, node_features, node_year, labels)

    def test_sample_query_nodes_is_deterministic_and_uses_valid_test_only(self) -> None:
        split_idx = {
            "train": list(range(0, 1000)),
            "valid": list(range(1000, 2500)),
            "test": list(range(2500, 4500)),
        }

        sample_a = sample_query_nodes(split_idx, seed=543, tier_sizes=TIER_SIZES)
        sample_b = sample_query_nodes(split_idx, seed=543, tier_sizes=TIER_SIZES)

        self.assertEqual(sample_a, sample_b)
        allowed = set(split_idx["valid"]) | set(split_idx["test"])
        disallowed = set(split_idx["train"])
        for tier, count in TIER_SIZES.items():
            self.assertEqual(len(sample_a[tier]), count)
            self.assertTrue(set(sample_a[tier]).issubset(allowed))
            self.assertFalse(set(sample_a[tier]) & disallowed)

    def test_build_split_assignments_counts_unassigned(self) -> None:
        assignments, counts = build_split_assignments(
            5,
            {
                "train": [0, 1],
                "valid": [2],
                "test": [3],
            },
        )

        self.assertEqual(assignments, ["train", "train", "valid", "test", "unassigned"])
        self.assertEqual(counts, {"train": 2, "valid": 1, "test": 1, "unassigned": 1})

    def test_write_query_manifests_generates_expected_counts_and_loads(self) -> None:
        split_idx = {
            "train": list(range(0, 1000)),
            "valid": list(range(1000, 2500)),
            "test": list(range(2500, 4500)),
        }
        tier_node_ids = sample_query_nodes(split_idx, seed=543, tier_sizes=TIER_SIZES)
        node_features = [[float(index), float(index) + 0.5] for index in range(4500)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_root = root / "queries"
            query_root.mkdir(parents=True)
            (query_root / "rag_retrieve.cql").write_text("RETURN 1", encoding="utf-8")
            (query_root / "baseline_vector_search.cql").write_text("RETURN 2", encoding="utf-8")
            (query_root / "baseline_traversal.cql").write_text("RETURN 3", encoding="utf-8")

            manifest_info = write_query_manifests(
                query_root=root / "manifests",
                tier_node_ids=tier_node_ids,
                node_features=node_features,
                index_name="paper_embedding_idx",
                scenarios=SCENARIOS,
            )

            self.assertEqual(manifest_info["smoke"]["count"], 1200)
            self.assertEqual(manifest_info["dev"]["count"], 6000)
            self.assertEqual(manifest_info["full"]["count"], 24000)

            cases = load_query_cases(root / "manifests" / "smoke_manifest.json", query_root)
            self.assertEqual(len(cases), 1200)
            self.assertEqual(cases[0].native_query_path, "rag_retrieve.cql")
            self.assertEqual(cases[0].baseline_vector_query_path, "baseline_vector_search.cql")
            self.assertEqual(cases[0].baseline_traversal_query_path, "baseline_traversal.cql")
            self.assertEqual(cases[0].params["index_name"], "paper_embedding_idx")


if __name__ == "__main__":
    unittest.main()
