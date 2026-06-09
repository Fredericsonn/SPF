import argparse
import csv
import heapq
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np

from subgraph_loader import load_subgraph


CSV_FIELDS = [
    "subgraph",
    "source_dataset",
    "node_count",
    "edge_count",
    "query_index",
    "source",
    "target",
    "heuristic_mode",
    "ml_scale",
    "geo_scale",
    "dijkstra_cost",
    "dijkstra_expanded",
    "dijkstra_runtime_ms",
    "ml_astar_cost",
    "ml_astar_expanded",
    "ml_astar_runtime_ms",
    "ml_astar_cost_ratio",
    "ml_astar_expanded_ratio",
    "optimal",
]


def geographic_distance_m(coords, node, target):
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

    return math.sqrt(dx**2 + dy**2)


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


def incoming_weight_stats(graph):
    incoming = {node: [] for node in graph}

    for _, neighbors in graph.items():
        for target, weight in neighbors.items():
            incoming[target].append(weight)

    return {
        node: (sum(weights) / len(weights) if weights else 0.0)
        for node, weights in incoming.items()
    }


def extract_feature_row(graph, coords, out_stats, in_avg, node, target):
    geodesic, abs_dx, abs_dy = coordinate_features_meters(coords, node, target)
    node_stats = out_stats[node]
    target_stats = out_stats[target]

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
        in_avg[target],
    ]


def compute_safe_geo_scale(graph, coords):
    scale = 1.0

    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            straight_line = geographic_distance_m(coords, source, target)

            if straight_line <= 0:
                continue

            scale = min(scale, weight / straight_line)

    return min(1.0, scale * 0.999)


def dijkstra_point_to_point(graph, source, target):
    distances = {node: math.inf for node in graph}
    distances[source] = 0.0
    queue = [(0.0, source)]
    expanded = 0
    visited = set()

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == target:
            return current_distance, expanded

        for neighbor, weight in graph[current].items():
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return math.inf, expanded


def make_heuristic_function(
    model,
    graph,
    coords,
    heuristic_mode,
    ml_scale,
    geo_scale,
):
    out_stats = outgoing_weight_stats(graph)
    in_avg = incoming_weight_stats(graph)
    cache = {}

    def predict_ml(node, target):
        key = (node, target)

        if key not in cache:
            features = extract_feature_row(
                graph=graph,
                coords=coords,
                out_stats=out_stats,
                in_avg=in_avg,
                node=node,
                target=target,
            )
            cache[key] = max(0.0, float(model.predict(np.asarray([features]))[0]))

        return cache[key]

    def heuristic(node, target):
        geo_h = geo_scale * geographic_distance_m(coords, node, target)
        ml_h = ml_scale * predict_ml(node, target)

        if heuristic_mode == "raw_ml":
            return ml_h

        if heuristic_mode == "min_ml_geo":
            return min(ml_h, geo_h)

        if heuristic_mode == "max_ml_geo":
            return max(ml_h, geo_h)

        if heuristic_mode == "geo_only":
            return geo_h

        raise ValueError(f"Unsupported heuristic mode: {heuristic_mode}")

    return heuristic


def ml_astar_point_to_point(graph, source, target, heuristic):
    g_score = {node: math.inf for node in graph}
    g_score[source] = 0.0
    queue = [(heuristic(source, target), source)]
    expanded = 0
    visited = set()

    while queue:
        _, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == target:
            return g_score[current], expanded

        for neighbor, weight in graph[current].items():
            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + heuristic(neighbor, target)
                heapq.heappush(queue, (priority, neighbor))

    return math.inf, expanded


def timed_call(function, *args):
    start = time.perf_counter()
    cost, expanded = function(*args)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return cost, expanded, runtime_ms


def sample_reachable_queries(graph, query_count, rng, max_attempts):
    nodes = sorted(graph)
    queries = []
    attempts = 0

    while len(queries) < query_count and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        cost, _ = dijkstra_point_to_point(graph, source, target)

        if math.isinf(cost):
            continue

        queries.append((source, target))

    if len(queries) < query_count:
        raise RuntimeError(
            f"Only sampled {len(queries)} reachable queries out of {query_count} "
            f"after {max_attempts} attempts."
        )

    return queries


def benchmark_subgraph(
    subgraph_path,
    model,
    query_count,
    seed,
    max_attempts,
    heuristic_mode,
    ml_scale,
):
    graph, coords, metadata = load_subgraph(subgraph_path)
    rng = random.Random(seed)
    queries = sample_reachable_queries(graph, query_count, rng, max_attempts)
    geo_scale = compute_safe_geo_scale(graph, coords)
    heuristic = make_heuristic_function(
        model=model,
        graph=graph,
        coords=coords,
        heuristic_mode=heuristic_mode,
        ml_scale=ml_scale,
        geo_scale=geo_scale,
    )
    rows = []

    print(
        f"Benchmarking ML-A* {metadata['name']} "
        f"({metadata['node_count']} nodes, {metadata['edge_count']} edges, "
        f"{query_count} queries, mode={heuristic_mode}, "
        f"ml_scale={ml_scale}, geo_scale={geo_scale:.6f})..."
    )

    for query_index, (source, target) in enumerate(queries, start=1):
        dijkstra_cost, dijkstra_expanded, dijkstra_runtime_ms = timed_call(
            dijkstra_point_to_point,
            graph,
            source,
            target,
        )
        ml_cost, ml_expanded, ml_runtime_ms = timed_call(
            ml_astar_point_to_point,
            graph,
            source,
            target,
            heuristic,
        )

        cost_ratio = ml_cost / dijkstra_cost if dijkstra_cost else math.inf
        expanded_ratio = (
            ml_expanded / dijkstra_expanded if dijkstra_expanded else math.inf
        )

        rows.append(
            {
                "subgraph": metadata["name"],
                "source_dataset": metadata["source_dataset"],
                "node_count": metadata["node_count"],
                "edge_count": metadata["edge_count"],
                "query_index": query_index,
                "source": source,
                "target": target,
                "heuristic_mode": heuristic_mode,
                "ml_scale": ml_scale,
                "geo_scale": geo_scale,
                "dijkstra_cost": dijkstra_cost,
                "dijkstra_expanded": dijkstra_expanded,
                "dijkstra_runtime_ms": dijkstra_runtime_ms,
                "ml_astar_cost": ml_cost,
                "ml_astar_expanded": ml_expanded,
                "ml_astar_runtime_ms": ml_runtime_ms,
                "ml_astar_cost_ratio": cost_ratio,
                "ml_astar_expanded_ratio": expanded_ratio,
                "optimal": ml_cost == dijkstra_cost,
            }
        )

        print(
            f"  query {query_index}/{query_count}: "
            f"{source}->{target}, "
            f"Dijkstra expanded={dijkstra_expanded}, "
            f"ML-A* expanded={ml_expanded}, "
            f"cost_ratio={cost_ratio:.6f}"
        )

    return rows


def collect_input_files(input_path, pattern):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched {input_path / pattern}")

    return files


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark A* using a trained ML heuristic model."
    )
    parser.add_argument("--model", required=True, help="Path to .joblib model bundle.")
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
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument(
        "--heuristic-mode",
        choices=["raw_ml", "min_ml_geo", "max_ml_geo", "geo_only"],
        default="raw_ml",
    )
    parser.add_argument(
        "--ml-scale",
        type=float,
        default=1.0,
        help="Scale applied to ML prediction before use as heuristic.",
    )

    args = parser.parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    input_files = collect_input_files(args.input, args.pattern)
    all_rows = []

    if isinstance(bundle, dict) and "metadata" in bundle:
        print("Loaded model metadata:")
        print(json.dumps(bundle["metadata"].get("metrics", {}), indent=2))

    for index, subgraph_path in enumerate(input_files, start=1):
        rows = benchmark_subgraph(
            subgraph_path=subgraph_path,
            model=model,
            query_count=args.queries,
            seed=args.seed + index,
            max_attempts=args.max_attempts,
            heuristic_mode=args.heuristic_mode,
            ml_scale=args.ml_scale,
        )
        all_rows.extend(rows)

    write_csv(all_rows, Path(args.output))
    print()
    print(f"Saved ML-A* benchmark results to {args.output}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
