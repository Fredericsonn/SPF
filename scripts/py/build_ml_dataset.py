import argparse
import heapq
import json
import math
import random
from pathlib import Path

import numpy as np

from subgraph_loader import load_subgraph


FEATURE_NAMES = [
    "geodesic_distance_m",
    "abs_dx_m",
    "abs_dy_m",
    "node_out_degree",
    "target_out_degree",
    "node_avg_out_weight",
    "target_avg_out_weight",
    "node_min_out_weight",
    "node_max_out_weight",
    "target_avg_in_weight",
]


def reverse_graph(graph):
    reversed_graph = {node: {} for node in graph}

    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            reversed_graph[target][source] = weight

    return reversed_graph


def incoming_weight_stats(graph):
    incoming = {node: [] for node in graph}

    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            incoming[target].append(weight)

    stats = {}
    for node, weights in incoming.items():
        if weights:
            stats[node] = sum(weights) / len(weights)
        else:
            stats[node] = 0.0

    return stats


def outgoing_weight_stats(graph):
    stats = {}

    for node, neighbors in graph.items():
        weights = list(neighbors.values())

        if weights:
            stats[node] = {
                "avg": sum(weights) / len(weights),
                "min": min(weights),
                "max": max(weights),
            }
        else:
            stats[node] = {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

    return stats


def dijkstra_all_distances(graph, start):
    distances = {node: math.inf for node in graph}
    distances[start] = 0.0
    queue = [(0.0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return distances


def coordinate_features_meters(coords, node, target):
    lon1_raw, lat1_raw = coords[node]
    lon2_raw, lat2_raw = coords[target]

    lon1 = math.radians(lon1_raw / 1_000_000)
    lat1 = math.radians(lat1_raw / 1_000_000)
    lon2 = math.radians(lon2_raw / 1_000_000)
    lat2 = math.radians(lat2_raw / 1_000_000)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    mean_lat = (lat1 + lat2) / 2

    dx = 6_371_000 * math.cos(mean_lat) * dlon
    dy = 6_371_000 * dlat
    geodesic = math.sqrt(dx**2 + dy**2)

    return geodesic, abs(dx), abs(dy)


def extract_feature_row(
    graph,
    coords,
    out_stats,
    target_avg_in_weight,
    node,
    target,
):
    node_stats = out_stats[node]
    target_stats = out_stats[target]
    geodesic, abs_dx, abs_dy = coordinate_features_meters(coords, node, target)

    return [
        geodesic,
        abs_dx,
        abs_dy,
        len(graph[node]),
        len(graph[target]),
        node_stats["avg"],
        target_stats["avg"],
        node_stats["min"],
        node_stats["max"],
        target_avg_in_weight[target],
    ]


def sample_targets(nodes, target_count, rng):
    if target_count > len(nodes):
        raise ValueError(
            f"Requested {target_count} targets, but graph only has {len(nodes)} nodes."
        )

    return rng.sample(nodes, target_count)


def sample_nodes_for_target(nodes, max_nodes_per_target, rng):
    if max_nodes_per_target is None or max_nodes_per_target >= len(nodes):
        return nodes

    return rng.sample(nodes, max_nodes_per_target)


def build_dataset_for_subgraph(
    subgraph_path,
    output_dir,
    target_count,
    max_nodes_per_target,
    seed,
):
    graph, coords, metadata = load_subgraph(subgraph_path)
    nodes = sorted(graph)
    rng = random.Random(seed)

    reversed_graph = reverse_graph(graph)
    out_stats = outgoing_weight_stats(graph)
    target_avg_in_weight = incoming_weight_stats(graph)
    targets = sample_targets(nodes, target_count, rng)

    X_parts = []
    y_parts = []
    node_id_parts = []
    target_id_parts = []

    print(
        f"Building ML dataset for {metadata['name']} "
        f"({len(nodes)} nodes, {metadata['edge_count']} edges, "
        f"{len(targets)} targets)..."
    )

    for target_index, target in enumerate(targets, start=1):
        distances_to_target = dijkstra_all_distances(reversed_graph, target)
        candidate_nodes = sample_nodes_for_target(nodes, max_nodes_per_target, rng)

        rows = []
        labels = []
        row_nodes = []
        row_targets = []

        for node in candidate_nodes:
            true_distance = distances_to_target[node]

            if math.isinf(true_distance):
                continue

            rows.append(
                extract_feature_row(
                    graph=graph,
                    coords=coords,
                    out_stats=out_stats,
                    target_avg_in_weight=target_avg_in_weight,
                    node=node,
                    target=target,
                )
            )
            labels.append(true_distance)
            row_nodes.append(node)
            row_targets.append(target)

        if rows:
            X_parts.append(np.asarray(rows, dtype=np.float32))
            y_parts.append(np.asarray(labels, dtype=np.float32))
            node_id_parts.append(np.asarray(row_nodes, dtype=np.int64))
            target_id_parts.append(np.asarray(row_targets, dtype=np.int64))

        print(
            f"  target {target_index}/{len(targets)}: "
            f"{target}, rows={len(rows)}"
        )

    if not X_parts:
        raise RuntimeError(f"No training rows generated for {subgraph_path}")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    node_ids = np.concatenate(node_id_parts)
    target_ids = np.concatenate(target_id_parts)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metadata['name']}_ml_dataset.npz"

    dataset_metadata = {
        "source_subgraph": str(subgraph_path),
        "subgraph_name": metadata["name"],
        "source_dataset": metadata["source_dataset"],
        "node_count": metadata["node_count"],
        "edge_count": metadata["edge_count"],
        "target_count": len(targets),
        "max_nodes_per_target": max_nodes_per_target,
        "row_count": int(X.shape[0]),
        "feature_names": FEATURE_NAMES,
        "seed": seed,
    }

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        node_ids=node_ids,
        target_ids=target_ids,
        targets=np.asarray(targets, dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
        metadata=json.dumps(dataset_metadata),
    )

    print(
        f"Saved {output_path} "
        f"({X.shape[0]} rows, {X.shape[1]} features)"
    )

    return output_path


def collect_input_files(input_path, pattern):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched {input_path / pattern}")

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Build supervised ML datasets for learned A* heuristics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Subgraph .json/.json.gz file or directory containing subgraphs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json.gz",
        help="Glob pattern when --input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where .npz ML datasets will be saved.",
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=50,
        help="Number of target nodes sampled per subgraph.",
    )
    parser.add_argument(
        "--max-nodes-per-target",
        type=int,
        default=None,
        help="Optional cap on sampled nodes per target. Defaults to all nodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for target/node sampling.",
    )

    args = parser.parse_args()
    input_files = collect_input_files(args.input, args.pattern)
    output_dir = Path(args.output_dir)

    for index, subgraph_path in enumerate(input_files, start=1):
        build_dataset_for_subgraph(
            subgraph_path=subgraph_path,
            output_dir=output_dir,
            target_count=args.targets,
            max_nodes_per_target=args.max_nodes_per_target,
            seed=args.seed + index,
        )


if __name__ == "__main__":
    main()
