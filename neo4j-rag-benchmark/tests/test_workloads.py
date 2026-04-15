from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.workloads import load_query_cases


class LoadQueryCasesTest(unittest.TestCase):
    def test_loads_enabled_manifest_entries_and_query_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_root = root / "queries"
            case_dir = query_root / "cases" / "q1"
            case_dir.mkdir(parents=True)
            (case_dir / "native.cql").write_text("RETURN 1", encoding="utf-8")
            (case_dir / "baseline_vector.cql").write_text("RETURN 2", encoding="utf-8")
            (case_dir / "baseline_traversal.cql").write_text("RETURN 3", encoding="utf-8")

            manifest = [
                {
                    "query_id": "q1",
                    "native_query": "cases/q1/native.cql",
                    "baseline_vector_query": "cases/q1/baseline_vector.cql",
                    "baseline_traversal_query": "cases/q1/baseline_traversal.cql",
                    "params": {"depth": 1},
                },
                {
                    "query_id": "q2",
                    "native_query": "cases/q1/native.cql",
                    "baseline_vector_query": "cases/q1/baseline_vector.cql",
                    "baseline_traversal_query": "cases/q1/baseline_traversal.cql",
                    "params": {"depth": 1},
                    "enabled": False,
                },
            ]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            cases = load_query_cases(manifest_path, query_root)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].query_id, "q1")
            self.assertEqual(cases[0].native_query.strip(), "RETURN 1")


if __name__ == "__main__":
    unittest.main()
