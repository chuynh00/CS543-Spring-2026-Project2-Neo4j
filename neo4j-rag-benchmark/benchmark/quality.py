from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmark.clients import BenchmarkClient, RetrievalRow
from benchmark.workloads import QueryCase


QUALITY_FIELDNAMES = [
    "query_id",
    "top_k",
    "depth",
    "native_seed_count",
    "baseline_seed_count",
    "native_mean_seed_ann_score",
    "baseline_mean_seed_ann_score",
    "seed_ann_score_difference",
    "native_mean_pairwise_neighborhood_overlap",
    "baseline_mean_pairwise_neighborhood_overlap",
    "neighborhood_overlap_difference",
    "native_row_count",
    "baseline_row_count",
    "row_count_difference",
    "row_count_ratio",
    "result_jaccard_overlap",
    "shared_result_count",
    "native_only_count",
    "baseline_only_count",
    "native_unique_label_count",
    "baseline_unique_label_count",
    "native_label_entropy",
    "baseline_label_entropy",
    "native_majority_label_fraction",
    "baseline_majority_label_fraction",
]


@dataclass(frozen=True)
class MethodReference:
    """Reference rows for one method/query pair."""

    rows: list[RetrievalRow]

    @property
    def node_ids(self) -> list[int]:
        return [row.node_id for row in self.rows]

    @property
    def seed_rows(self) -> list[RetrievalRow]:
        return [row for row in self.rows if row.hop_depth == 0]

    @property
    def seed_ids(self) -> list[int]:
        return [row.node_id for row in self.seed_rows]

    @property
    def seed_scores(self) -> list[float]:
        return [float(row.score) for row in self.seed_rows if row.score is not None]


def mean(values: Iterable[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def jaccard(left: set[int], right: set[int]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def result_jaccard_overlap(native_ids: Iterable[int], baseline_ids: Iterable[int]) -> float | None:
    return jaccard(set(native_ids), set(baseline_ids))


def mean_pairwise_neighborhood_overlap(seed_ids: list[int], neighborhoods: dict[int, set[int]]) -> float | None:
    if len(seed_ids) < 2:
        return None

    overlaps: list[float] = []
    for left_index, left_seed in enumerate(seed_ids):
        for right_seed in seed_ids[left_index + 1 :]:
            overlap = jaccard(neighborhoods.get(left_seed, set()), neighborhoods.get(right_seed, set()))
            if overlap is not None:
                overlaps.append(overlap)

    return mean(overlaps)


def label_entropy(labels: Iterable[int]) -> float | None:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return None

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def majority_label_fraction(labels: Iterable[int]) -> float | None:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return None
    return max(counts.values()) / total


def label_diversity(node_ids: Iterable[int], labels_by_node: dict[int, int]) -> dict[str, float | int | None]:
    labels = [labels_by_node[node_id] for node_id in node_ids if node_id in labels_by_node]
    if not labels:
        return {
            "unique_label_count": None,
            "label_entropy": None,
            "majority_label_fraction": None,
        }

    return {
        "unique_label_count": len(set(labels)),
        "label_entropy": label_entropy(labels),
        "majority_label_fraction": majority_label_fraction(labels),
    }


def extract_top_k_depth(query_case: QueryCase) -> tuple[int | None, int | None]:
    top_k = query_case.params.get("top_k")
    depth = query_case.params.get("depth")
    return int(top_k) if top_k is not None else None, int(depth) if depth is not None else None


def build_quality_row(
    *,
    query_case: QueryCase,
    native: MethodReference,
    baseline: MethodReference,
    native_neighborhoods: dict[int, set[int]],
    baseline_neighborhoods: dict[int, set[int]],
    labels_by_node: dict[int, int],
) -> dict[str, Any]:
    top_k, depth = extract_top_k_depth(query_case)

    native_node_set = set(native.node_ids)
    baseline_node_set = set(baseline.node_ids)
    shared_result_count = len(native_node_set & baseline_node_set)

    native_mean_seed_score = mean(native.seed_scores)
    baseline_mean_seed_score = mean(baseline.seed_scores)
    native_overlap = mean_pairwise_neighborhood_overlap(native.seed_ids, native_neighborhoods)
    baseline_overlap = mean_pairwise_neighborhood_overlap(baseline.seed_ids, baseline_neighborhoods)
    native_labels = label_diversity(native.node_ids, labels_by_node)
    baseline_labels = label_diversity(baseline.node_ids, labels_by_node)

    return {
        "query_id": query_case.query_id,
        "top_k": top_k,
        "depth": depth,
        "native_seed_count": len(native.seed_ids),
        "baseline_seed_count": len(baseline.seed_ids),
        "native_mean_seed_ann_score": native_mean_seed_score,
        "baseline_mean_seed_ann_score": baseline_mean_seed_score,
        "seed_ann_score_difference": difference(native_mean_seed_score, baseline_mean_seed_score),
        "native_mean_pairwise_neighborhood_overlap": native_overlap,
        "baseline_mean_pairwise_neighborhood_overlap": baseline_overlap,
        "neighborhood_overlap_difference": difference(native_overlap, baseline_overlap),
        "native_row_count": len(native.node_ids),
        "baseline_row_count": len(baseline.node_ids),
        "row_count_difference": len(native.node_ids) - len(baseline.node_ids),
        "row_count_ratio": ratio(len(native.node_ids), len(baseline.node_ids)),
        "result_jaccard_overlap": result_jaccard_overlap(native.node_ids, baseline.node_ids),
        "shared_result_count": shared_result_count,
        "native_only_count": len(native_node_set - baseline_node_set),
        "baseline_only_count": len(baseline_node_set - native_node_set),
        "native_unique_label_count": native_labels["unique_label_count"],
        "baseline_unique_label_count": baseline_labels["unique_label_count"],
        "native_label_entropy": native_labels["label_entropy"],
        "baseline_label_entropy": baseline_labels["label_entropy"],
        "native_majority_label_fraction": native_labels["majority_label_fraction"],
        "baseline_majority_label_fraction": baseline_labels["majority_label_fraction"],
    }


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_quality_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in QUALITY_FIELDNAMES})


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [field for field in QUALITY_FIELDNAMES if field not in {"query_id", "top_k", "depth"}]
    return {
        "query_count": len(rows),
        "pairwise_seed_metric_count": sum(
            1
            for row in rows
            if row.get("native_mean_pairwise_neighborhood_overlap") is not None
            and row.get("baseline_mean_pairwise_neighborhood_overlap") is not None
        ),
        "means": {field: mean(row.get(field) for row in rows) for field in numeric_fields},
    }


def grouped_aggregates(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    return {str(group): aggregate_rows(group_rows) for group, group_rows in sorted(grouped.items(), key=lambda item: str(item[0]))}


def build_quality_summary(*, session_id: str, rows: list[dict[str, Any]], quality_csv_path: Path) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "quality_csv_path": str(quality_csv_path),
        "overall": aggregate_rows(rows),
        "by_top_k": grouped_aggregates(rows, "top_k"),
        "by_depth": grouped_aggregates(rows, "depth"),
    }


def collect_quality_rows(
    *,
    query_cases: list[QueryCase],
    native_rows_by_query: dict[str, list[RetrievalRow]],
    baseline_rows_by_query: dict[str, list[RetrievalRow]],
    client: BenchmarkClient,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    neighborhood_cache: dict[int, set[int]] = {}
    label_cache: dict[int, int] = {}
    for query_case in query_cases:
        native = MethodReference(native_rows_by_query.get(query_case.query_id, []))
        baseline = MethodReference(baseline_rows_by_query.get(query_case.query_id, []))
        all_seed_ids = sorted(set(native.seed_ids) | set(baseline.seed_ids))
        all_node_ids = sorted(set(native.node_ids) | set(baseline.node_ids))

        missing_seed_ids = [seed_id for seed_id in all_seed_ids if seed_id not in neighborhood_cache]
        if missing_seed_ids:
            neighborhood_cache.update(client.fetch_one_hop_neighborhoods(missing_seed_ids))

        missing_node_ids = [node_id for node_id in all_node_ids if node_id not in label_cache]
        if missing_node_ids:
            label_cache.update(client.fetch_node_labels(missing_node_ids))

        rows.append(
            build_quality_row(
                query_case=query_case,
                native=native,
                baseline=baseline,
                native_neighborhoods={seed_id: neighborhood_cache.get(seed_id, set()) for seed_id in native.seed_ids},
                baseline_neighborhoods={seed_id: neighborhood_cache.get(seed_id, set()) for seed_id in baseline.seed_ids},
                labels_by_node=label_cache,
            )
        )
    return rows


def write_comparison_quality_artifacts(
    *,
    query_cases: list[QueryCase],
    native_rows_by_query: dict[str, list[RetrievalRow]],
    baseline_rows_by_query: dict[str, list[RetrievalRow]],
    client: BenchmarkClient,
    results_root: Path,
    session_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    rows = collect_quality_rows(
        query_cases=query_cases,
        native_rows_by_query=native_rows_by_query,
        baseline_rows_by_query=baseline_rows_by_query,
        client=client,
    )
    quality_csv_path = results_root / "eval" / f"comparison_quality-{session_id}.csv"
    quality_summary_path = results_root / "summaries" / f"comparison_quality-{session_id}.json"
    write_quality_csv(quality_csv_path, rows)
    summary = build_quality_summary(session_id=session_id, rows=rows, quality_csv_path=quality_csv_path)
    quality_summary_path.parent.mkdir(parents=True, exist_ok=True)
    quality_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return quality_csv_path, quality_summary_path, summary
