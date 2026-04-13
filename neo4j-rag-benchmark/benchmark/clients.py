from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import Driver, GraphDatabase


@dataclass(frozen=True)
class RetrievalRow:
    """One row returned by a retrieval method, normalized for comparison."""

    node_id: int
    hop_depth: int
    score: float | None


class BenchmarkClient:
    """Neo4j driver wrapper for the native procedure and the two-call baseline."""

    def __init__(self, driver: Driver, query_dir: Path, database: str):
        self._driver = driver
        self._database = database
        self._native_query = (query_dir / "rag_retrieve.cql").read_text(encoding="utf-8")
        self._vector_query = (query_dir / "baseline_vector_search.cql").read_text(encoding="utf-8")
        self._traversal_template = (query_dir / "baseline_traversal.cql").read_text(encoding="utf-8")

    @classmethod
    def connect(cls, uri: str, user: str, password: str, query_dir: Path, database: str) -> "BenchmarkClient":
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver, query_dir=query_dir, database=database)

    def close(self) -> None:
        self._driver.close()

    def run_native_rag(
        self,
        *,
        index_name: str,
        embedding: list[float],
        top_k: int,
        depth: int,
        config: dict[str, Any],
    ) -> list[RetrievalRow]:
        with self._driver.session(database=self._database) as session:
            records = session.run(
                self._native_query,
                index_name=index_name,
                embedding=embedding,
                top_k=top_k,
                depth=depth,
                config=config,
            )
            return [
                RetrievalRow(
                    node_id=int(record["nodeId"]),
                    hop_depth=int(record["hopDepth"]),
                    score=float(record["score"]),
                )
                for record in records
            ]

    def run_two_call_baseline(
        self,
        *,
        index_name: str,
        embedding: list[float],
        top_k: int,
        depth: int,
    ) -> tuple[list[RetrievalRow], list[int]]:
        with self._driver.session(database=self._database) as session:
            seed_records = list(
                session.run(
                    self._vector_query,
                    index_name=index_name,
                    embedding=embedding,
                    top_k=top_k,
                )
            )
            seed_ids = [int(record["nodeId"]) for record in seed_records]
            seed_scores = {int(record["nodeId"]): float(record["score"]) for record in seed_records}

            if not seed_ids:
                return [], []

            traversal_query = self._traversal_template.replace("__DEPTH__", str(depth))
            rows = [
                RetrievalRow(
                    node_id=int(record["nodeId"]),
                    hop_depth=int(record["hopDepth"]),
                    score=seed_scores.get(int(record["nodeId"])),
                )
                for record in session.run(traversal_query, seed_ids=seed_ids)
            ]
            return rows, seed_ids
