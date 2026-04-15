from __future__ import annotations

from dataclasses import dataclass
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

    def __init__(self, driver: Driver, database: str):
        self._driver = driver
        self._database = database

    @classmethod
    def connect(cls, uri: str, user: str, password: str, database: str) -> "BenchmarkClient":
        if bool(user) != bool(password):
            raise ValueError("Neo4j user and password must either both be set or both be omitted.")
        driver = GraphDatabase.driver(uri, auth=(user, password)) if user else GraphDatabase.driver(uri)
        return cls(driver, database=database)

    def close(self) -> None:
        self._driver.close()

    def run_native_rag(
        self,
        *,
        query_text: str,
        params: dict[str, Any],
        config: dict[str, Any],
    ) -> list[RetrievalRow]:
        execution_params = dict(params)
        execution_params["config"] = config

        with self._driver.session(database=self._database) as session:
            records = session.run(query_text, **execution_params)
            return [
                RetrievalRow(
                    node_id=int(record["nodeId"]),
                    hop_depth=int(record["hopDepth"]),
                    score=float(record["score"]) if record["score"] is not None else None,
                )
                for record in records
            ]

    def run_two_call_baseline(
        self,
        *,
        vector_query_text: str,
        traversal_query_text: str,
        params: dict[str, Any],
    ) -> tuple[list[RetrievalRow], list[int]]:
        with self._driver.session(database=self._database) as session:
            seed_records = list(session.run(vector_query_text, **params))
            seed_ids = [int(record["nodeId"]) for record in seed_records]
            seed_scores = {int(record["nodeId"]): float(record["score"]) for record in seed_records}

            if not seed_ids:
                return [], []

            rendered_traversal_query = traversal_query_text.replace("__DEPTH__", str(params["depth"]))
            traversal_params = dict(params)
            traversal_params["seed_ids"] = seed_ids
            rows = [
                RetrievalRow(
                    node_id=int(record["nodeId"]),
                    hop_depth=int(record["hopDepth"]),
                    score=seed_scores.get(int(record["nodeId"])),
                )
                for record in session.run(rendered_traversal_query, **traversal_params)
            ]
            return rows, seed_ids
