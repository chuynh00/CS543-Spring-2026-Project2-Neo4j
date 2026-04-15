from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "ogbn-arxiv"
EXPECTED_NODE_COUNT = 169_343
EXPECTED_EDGE_COUNT = 1_166_243
EMBEDDING_DIMENSION = 128
DEFAULT_INDEX_NAME = "paper_embedding_idx"
QUERY_SEED = 543
TIER_SIZES = {
    "smoke": 100,
    "dev": 500,
    "full": 2_000,
}
SCENARIOS = tuple((top_k, depth) for top_k in (1, 3, 5, 10) for depth in (0, 1, 2))
SHARED_QUERY_FILES = {
    "native_query": "rag_retrieve.cql",
    "baseline_vector_query": "baseline_vector_search.cql",
    "baseline_traversal_query": "baseline_traversal.cql",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ogbn-arxiv and export Neo4j benchmark artifacts.")
    parser.add_argument(
        "--raw-root",
        default=str(ROOT / "data" / "raw" / "ogb"),
        help="Directory where the OGB loader caches downloaded source data.",
    )
    parser.add_argument(
        "--processed-root",
        default=str(ROOT / "data" / "processed" / DATASET_NAME),
        help="Directory for processed Neo4j import CSVs and the prep manifest.",
    )
    parser.add_argument(
        "--query-root",
        default=str(ROOT / "data" / "query_sets" / "ogbn_arxiv"),
        help="Directory for generated benchmark manifests.",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Vector index name embedded into the generated benchmark manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=QUERY_SEED,
        help="Deterministic random seed used for query sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    processed_root = Path(args.processed_root).resolve()
    query_root = Path(args.query_root).resolve()

    processed_root.mkdir(parents=True, exist_ok=True)
    query_root.mkdir(parents=True, exist_ok=True)

    dataset, split_idx, np = load_dataset(raw_root)
    graph, labels = dataset[0]

    node_features = np.asarray(graph["node_feat"], dtype=np.float32)
    num_nodes = int(graph.get("num_nodes", node_features.shape[0]))
    edge_index = np.asarray(graph["edge_index"])
    node_year = load_node_year(graph, raw_root, np)
    label_array = np.asarray(labels).reshape(-1)

    validate_dataset_shape(num_nodes, edge_index, node_features, node_year, label_array)

    split_assignments, split_counts = build_split_assignments(num_nodes, split_idx)
    papers_path, papers_hash = write_papers_csv(
        processed_root / "papers.csv",
        node_year=node_year,
        labels=label_array,
        splits=split_assignments,
        node_features=node_features,
    )
    cites_path, cites_hash = write_cites_csv(processed_root / "cites.csv", edge_index=edge_index)

    tier_node_ids = sample_query_nodes(split_idx, seed=args.seed, tier_sizes=TIER_SIZES)
    query_manifests = write_query_manifests(
        query_root=query_root,
        tier_node_ids=tier_node_ids,
        node_features=node_features,
        index_name=args.index_name,
        scenarios=SCENARIOS,
    )

    manifest_core = {
        "dataset_name": DATASET_NAME,
        "expected_counts": {
            "node_count": EXPECTED_NODE_COUNT,
            "edge_count": EXPECTED_EDGE_COUNT,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
        "node_count": num_nodes,
        "edge_count": int(edge_index.shape[1]),
        "embedding_dimension": int(node_features.shape[1]),
        "index_name": args.index_name,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "query_root": str(query_root),
        "split_counts": split_counts,
        "query_seed": args.seed,
        "tier_sizes": TIER_SIZES,
        "scenarios": [{"top_k": top_k, "depth": depth} for top_k, depth in SCENARIOS],
        "files": {
            "papers_csv": {
                "path": relativize(papers_path),
                "sha256": papers_hash,
            },
            "cites_csv": {
                "path": relativize(cites_path),
                "sha256": cites_hash,
            },
            "query_manifests": query_manifests,
        },
    }

    manifest = {
        **manifest_core,
        "dataset_fingerprint": hash_json(manifest_core),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = processed_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote dataset manifest to {manifest_path}")
    print(f"Wrote node CSV to {papers_path}")
    print(f"Wrote relationship CSV to {cites_path}")
    print(f"Wrote query manifests under {query_root}")


def load_dataset(raw_root: Path):
    try:
        import numpy as np
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:  # pragma: no cover - exercised only in the user environment
        raise SystemExit(
            "Missing Python dependencies for ogbn-arxiv prep. Run ./scripts/bootstrap_python.sh first."
        ) from exc

    # OGB still calls torch.load() without an explicit weights_only argument.
    # Newer PyTorch defaults that to True, which breaks loading these trusted
    # preprocessed dataset artifacts unless we opt back into the legacy behavior.
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0") != "1" and "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    dataset = NodePropPredDataset(name=DATASET_NAME, root=str(raw_root))
    split_idx = dataset.get_idx_split()
    return dataset, split_idx, np


def load_node_year(graph: dict[str, Any], raw_root: Path, np):
    if "node_year" in graph:
        return np.asarray(graph["node_year"]).reshape(-1)

    for candidate in sorted(raw_root.rglob("node_year*")):
        if candidate.suffix == ".npy":
            return np.asarray(np.load(candidate)).reshape(-1)
        if candidate.suffix == ".npz":
            with np.load(candidate) as archive:
                for key in archive.files:
                    return np.asarray(archive[key]).reshape(-1)

    raise ValueError("Could not locate node_year data for ogbn-arxiv.")


def validate_dataset_shape(num_nodes: int, edge_index, node_features, node_year, labels) -> None:
    if num_nodes != EXPECTED_NODE_COUNT:
        raise ValueError(f"Unexpected ogbn-arxiv node count: {num_nodes} != {EXPECTED_NODE_COUNT}")
    if int(edge_index.shape[1]) != EXPECTED_EDGE_COUNT:
        raise ValueError(f"Unexpected ogbn-arxiv edge count: {edge_index.shape[1]} != {EXPECTED_EDGE_COUNT}")
    if int(node_features.shape[0]) != num_nodes:
        raise ValueError("Node feature row count does not match num_nodes.")
    if int(node_features.shape[1]) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected ogbn-arxiv embedding dimension: {node_features.shape[1]} != {EMBEDDING_DIMENSION}"
        )
    if len(node_year) != num_nodes:
        raise ValueError("node_year length does not match num_nodes.")
    if len(labels) != num_nodes:
        raise ValueError("Label array length does not match num_nodes.")


def build_split_assignments(num_nodes: int, split_idx: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    assignments = ["unassigned"] * num_nodes
    split_counts = {"train": 0, "valid": 0, "test": 0, "unassigned": 0}

    for split_name, raw_indices in split_idx.items():
        normalized_name = "valid" if split_name == "valid" else split_name
        indices = [int(value) for value in raw_indices]
        for index in indices:
            assignments[index] = normalized_name
        split_counts[normalized_name] = len(indices)

    split_counts["unassigned"] = assignments.count("unassigned")
    return assignments, split_counts


def write_papers_csv(
    path: Path,
    *,
    node_year,
    labels,
    splits: list[str],
    node_features,
) -> tuple[Path, str]:
    hasher = hashlib.sha256()
    header = "paperId:ID(Paper-ID),year:int,label:int,split,embedding:float[]\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        write_line(handle, hasher, header)
        for node_id in range(len(splits)):
            embedding = ";".join(f"{float(value):.9g}" for value in node_features[node_id])
            line = (
                f"{node_id},{int(node_year[node_id])},{int(labels[node_id])},"
                f"{splits[node_id]},{embedding}\n"
            )
            write_line(handle, hasher, line)

    return path, hasher.hexdigest()


def write_cites_csv(path: Path, *, edge_index) -> tuple[Path, str]:
    hasher = hashlib.sha256()
    header = ":START_ID(Paper-ID),:END_ID(Paper-ID)\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        write_line(handle, hasher, header)
        for source, target in zip(edge_index[0], edge_index[1], strict=True):
            write_line(handle, hasher, f"{int(source)},{int(target)}\n")

    return path, hasher.hexdigest()


def sample_query_nodes(
    split_idx: dict[str, Any],
    *,
    seed: int,
    tier_sizes: dict[str, int],
) -> dict[str, list[int]]:
    candidate_ids = [int(value) for split_name in ("valid", "test") for value in split_idx[split_name]]
    if len(candidate_ids) < tier_sizes["full"]:
        raise ValueError(
            f"Need at least {tier_sizes['full']} candidate nodes for the full tier, found {len(candidate_ids)}."
        )

    rng = random.Random(seed)
    full_sample = rng.sample(candidate_ids, tier_sizes["full"])
    return {tier: full_sample[:count] for tier, count in tier_sizes.items()}


def write_query_manifests(
    *,
    query_root: Path,
    tier_node_ids: dict[str, list[int]],
    node_features,
    index_name: str,
    scenarios: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    query_manifests: dict[str, Any] = {}
    query_root.mkdir(parents=True, exist_ok=True)

    for tier, node_ids in tier_node_ids.items():
        cases: list[dict[str, Any]] = []
        for node_id in node_ids:
            embedding = [float(value) for value in node_features[node_id]]
            for top_k, depth in scenarios:
                cases.append(
                    {
                        "query_id": f"paper-{node_id}-topk-{top_k}-depth-{depth}",
                        **SHARED_QUERY_FILES,
                        "params": {
                            "index_name": index_name,
                            "embedding": embedding,
                            "top_k": top_k,
                            "depth": depth,
                        },
                    }
                )

        path = query_root / f"{tier}_manifest.json"
        payload = json.dumps(cases, indent=2)
        path.write_text(payload, encoding="utf-8")
        query_manifests[tier] = {
            "path": relativize(path),
            "count": len(cases),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    return query_manifests


def write_line(handle, hasher: hashlib._Hash, line: str) -> None:
    handle.write(line)
    hasher.update(line.encode("utf-8"))


def relativize(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
