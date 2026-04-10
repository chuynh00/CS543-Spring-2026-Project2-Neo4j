from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryCase:
    """One benchmark query input loaded from a JSONL workload file."""

    query_id: str
    embedding: list[float]
    top_k: int
    depth: int
    config: dict[str, Any]



def load_query_cases(path: str | Path) -> list[QueryCase]:
    """Load a JSONL query set into typed benchmark cases."""

    query_path = Path(path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query set not found: {query_path}")

    cases: list[QueryCase] = []
    with query_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw = json.loads(stripped)
            cases.append(
                QueryCase(
                    query_id=str(raw["query_id"]),
                    embedding=[float(value) for value in raw["embedding"]],
                    top_k=int(raw["top_k"]),
                    depth=int(raw["depth"]),
                    config=dict(raw.get("config", {})),
                )
            )

    if not cases:
        raise ValueError(f"No benchmark queries found in {query_path}")

    return cases
