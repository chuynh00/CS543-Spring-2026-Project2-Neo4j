from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryCase:
    """One benchmark case loaded from a manifest entry and its backing query files."""

    query_id: str
    native_query_path: str
    baseline_vector_query_path: str
    baseline_traversal_query_path: str
    native_query: str
    baseline_vector_query: str
    baseline_traversal_query: str
    params: dict[str, Any]
    native_config: dict[str, Any]



def _load_query_text(query_root: Path, relative_path: str) -> tuple[str, str]:
    query_path = query_root / relative_path
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    return relative_path, query_path.read_text(encoding="utf-8")



def load_query_cases(path: str | Path, query_root: str | Path) -> list[QueryCase]:
    """Load a JSON manifest and its per-case query files into typed benchmark cases."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Query manifest not found: {manifest_path}")

    query_root_path = Path(query_root)
    raw_cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError(f"Query manifest must be a JSON array: {manifest_path}")

    cases: list[QueryCase] = []
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest entry {index} must be a JSON object")
        if not raw.get("enabled", True):
            continue

        query_id = str(raw["query_id"])
        native_query_path, native_query = _load_query_text(query_root_path, str(raw["native_query"]))
        baseline_vector_query_path, baseline_vector_query = _load_query_text(
            query_root_path, str(raw["baseline_vector_query"])
        )
        baseline_traversal_query_path, baseline_traversal_query = _load_query_text(
            query_root_path, str(raw["baseline_traversal_query"])
        )

        params = dict(raw.get("params", {}))
        if not params:
            raise ValueError(f"Manifest entry {query_id} is missing params")

        cases.append(
            QueryCase(
                query_id=query_id,
                native_query_path=native_query_path,
                baseline_vector_query_path=baseline_vector_query_path,
                baseline_traversal_query_path=baseline_traversal_query_path,
                native_query=native_query,
                baseline_vector_query=baseline_vector_query,
                baseline_traversal_query=baseline_traversal_query,
                params=params,
                native_config=dict(raw.get("native_config", {})),
            )
        )

    if not cases:
        raise ValueError(f"No enabled benchmark queries found in {manifest_path}")

    return cases
