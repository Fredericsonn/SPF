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
    "heuristic_calls",
    "ml_chosen_count",
    "geo_chosen_count",
    "equal_chosen_count",
    "ml_chosen_ratio",
    "geo_chosen_ratio",
    "mean_ml_heuristic",
    "mean_geo_heuristic",
    "mean_chosen_heuristic",
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


class HeuristicTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.ml_chosen = 0
        self.geo_chosen = 0
        self.equal_chosen = 0
        self.ml_sum = 0.0
        self.geo_sum = 0.0
        self.chosen_sum = 0.0

    def record(self, ml_h, geo_h, chosen_h, source):
        self.calls += 1
        self.ml_sum += ml_h
        self.geo_sum += geo_h
        self.chosen_sum += chosen_h

        if source == "ml":
            self.ml_chosen += 1
        elif source == "geo":
            self.geo_chosen += 1
        elif source == "equal":
            self.equal_chosen += 1
        else:
            raise ValueError(f"Unsupported heuristic source: {source}")

    def snapshot(self):
        if self.calls == 0:
            return {
                "heuristic_calls": 0,
                "ml_chosen_count": 0,
                "geo_chosen_count": 0,
                "equal_chosen_count": 0,
                "ml_chosen_ratio": 0.0,
                "geo_chosen_ratio": 0.0,
                "mean_ml_heuristic": 0.0,
                "mean_geo_heuristic": 0.0,
                "mean_chosen_heuristic": 0.0,
            }

        return {
            "heuristic_calls": self.calls,
            "ml_chosen_count": self.ml_chosen,
            "geo_chosen_count": self.geo_chosen,
            "equal_chosen_count": self.equal_chosen,
            "ml_chosen_ratio": self.ml_chosen / self.calls,
            "geo_chosen_ratio": self.geo_chosen / self.calls,
            "mean_ml_heuristic": self.ml_sum / self.calls,
            "mean_geo_heuristic": self.geo_sum / self.calls,
            "mean_chosen_heuristic": self.chosen_sum / self.calls,
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


def safe_geo_astar_point_to_point(graph, coords, source, target, geo_scale):
    g_score = {node: math.inf for node in graph}
    g_score[source] = 0.0
    queue = [(geo_scale * geographic_distance_m(coords, source, target), source)]
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
                priority = (
                    tentative_g
                    + geo_scale * geographic_distance_m(coords, neighbor, target)
                )
                heapq.heappush(queue, (priority, neighbor))

    return math.inf, expanded


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


def traversal_cache_path(cache_dir, metadata, query_count, seed):
    return Path(cache_dir) / f"{metadata['name']}_q{query_count}_seed{seed}.csv"


def load_traversal_cache(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "query_index": int(row["query_index"]),
                    "source": int(row["source"]),
                    "target": int(row["target"]),
                    "dijkstra_cost": float(row["dijkstra_cost"]),
                    "dijkstra_expanded": int(row["dijkstra_expanded"]),
                    "dijkstra_runtime_ms": float(row["dijkstra_runtime_ms"]),
                    "safe_astar_cost": float(row["safe_astar_cost"]),
                    "safe_astar_expanded": int(row["safe_astar_expanded"]),
                    "safe_astar_runtime_ms": float(row["safe_astar_runtime_ms"]),
                    "safe_astar_cost_ratio": float(row["safe_astar_cost_ratio"]),
                    "safe_astar_expanded_ratio": float(row["safe_astar_expanded_ratio"]),
                    "safe_astar_heuristic_scale": float(
                        row["safe_astar_heuristic_scale"]
                    ),
                }
            )

    return rows


def write_traversal_cache(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_index",
        "source",
        "target",
        "dijkstra_cost",
        "dijkstra_expanded",
        "dijkstra_runtime_ms",
        "safe_astar_cost",
        "safe_astar_expanded",
        "safe_astar_runtime_ms",
        "safe_astar_cost_ratio",
        "safe_astar_expanded_ratio",
        "safe_astar_heuristic_scale",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_traversal_cache_rows(graph, coords, query_count, seed, max_attempts, geo_scale):
    rng = random.Random(seed)
    nodes = sorted(graph)
    rows = []
    attempts = 0

    while len(rows) < query_count and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        dijkstra_cost, dijkstra_expanded, dijkstra_runtime_ms = timed_call(
            dijkstra_point_to_point,
            graph,
            source,
            target,
        )

        if math.isinf(dijkstra_cost):
            continue

        safe_cost, safe_expanded, safe_runtime_ms = timed_call(
            safe_geo_astar_point_to_point,
            graph,
            coords,
            source,
            target,
            geo_scale,
        )

        rows.append(
            {
                "query_index": len(rows) + 1,
                "source": source,
                "target": target,
                "dijkstra_cost": dijkstra_cost,
                "dijkstra_expanded": dijkstra_expanded,
                "dijkstra_runtime_ms": dijkstra_runtime_ms,
                "safe_astar_cost": safe_cost,
                "safe_astar_expanded": safe_expanded,
                "safe_astar_runtime_ms": safe_runtime_ms,
                "safe_astar_cost_ratio": (
                    safe_cost / dijkstra_cost if dijkstra_cost else math.inf
                ),
                "safe_astar_expanded_ratio": (
                    safe_expanded / dijkstra_expanded
                    if dijkstra_expanded
                    else math.inf
                ),
                "safe_astar_heuristic_scale": geo_scale,
            }
        )

        print(
            f"  cached query {len(rows)}/{query_count}: "
            f"{source}->{target}, "
            f"Dijkstra expanded={dijkstra_expanded}, "
            f"safe A* expanded={safe_expanded}",
            flush=True,
        )

    if len(rows) < query_count:
        raise RuntimeError(
            f"Only sampled {len(rows)} reachable queries out of {query_count} "
            f"after {max_attempts} attempts."
        )

    return rows


def get_traversal_rows(
    graph,
    coords,
    metadata,
    query_count,
    seed,
    max_attempts,
    geo_scale,
    cache_dir,
):
    if cache_dir is None:
        return build_traversal_cache_rows(
            graph=graph,
            coords=coords,
            query_count=query_count,
            seed=seed,
            max_attempts=max_attempts,
            geo_scale=geo_scale,
        )

    path = traversal_cache_path(cache_dir, metadata, query_count, seed)

    if path.exists():
        print(f"Loading traversal cache: {path}", flush=True)
        return load_traversal_cache(path)

    print(f"Creating traversal cache: {path}", flush=True)
    rows = build_traversal_cache_rows(
        graph=graph,
        coords=coords,
        query_count=query_count,
        seed=seed,
        max_attempts=max_attempts,
        geo_scale=geo_scale,
    )
    write_traversal_cache(path, rows)
    print(f"Saved traversal cache: {path}", flush=True)
    return rows


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
    tracker = HeuristicTracker()

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
            tracker.record(ml_h, geo_h, ml_h, "ml")
            return ml_h

        if heuristic_mode == "min_ml_geo":
            if ml_h < geo_h:
                tracker.record(ml_h, geo_h, ml_h, "ml")
                return ml_h
            if geo_h < ml_h:
                tracker.record(ml_h, geo_h, geo_h, "geo")
                return geo_h

            tracker.record(ml_h, geo_h, ml_h, "equal")
            return ml_h

        if heuristic_mode == "max_ml_geo":
            if ml_h > geo_h:
                tracker.record(ml_h, geo_h, ml_h, "ml")
                return ml_h
            if geo_h > ml_h:
                tracker.record(ml_h, geo_h, geo_h, "geo")
                return geo_h

            tracker.record(ml_h, geo_h, ml_h, "equal")
            return ml_h

        if heuristic_mode == "geo_only":
            tracker.record(ml_h, geo_h, geo_h, "geo")
            return geo_h

        raise ValueError(f"Unsupported heuristic mode: {heuristic_mode}")

    return heuristic, tracker


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
    traversal_cache_dir,
):
    graph, coords, metadata = load_subgraph(subgraph_path)
    geo_scale = compute_safe_geo_scale(graph, coords)
    traversal_rows = get_traversal_rows(
        graph=graph,
        coords=coords,
        metadata=metadata,
        query_count=query_count,
        seed=seed,
        max_attempts=max_attempts,
        geo_scale=geo_scale,
        cache_dir=traversal_cache_dir,
    )
    heuristic, tracker = make_heuristic_function(
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
        f"{len(traversal_rows)} cached queries, mode={heuristic_mode}, "
        f"ml_scale={ml_scale}, geo_scale={geo_scale:.6f})..."
    )

    for traversal in traversal_rows:
        query_index = traversal["query_index"]
        source = traversal["source"]
        target = traversal["target"]
        dijkstra_cost = traversal["dijkstra_cost"]
        dijkstra_expanded = traversal["dijkstra_expanded"]
        dijkstra_runtime_ms = traversal["dijkstra_runtime_ms"]

        tracker.reset()
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
        heuristic_stats = tracker.snapshot()

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
                **heuristic_stats,
            }
        )

        print(
            f"  query {query_index}/{query_count}: "
            f"{source}->{target}, "
            f"Dijkstra expanded={dijkstra_expanded}, "
            f"ML-A* expanded={ml_expanded}, "
            f"cost_ratio={cost_ratio:.6f}, "
            f"ml_chosen={heuristic_stats['ml_chosen_ratio']:.2%}, "
            f"geo_chosen={heuristic_stats['geo_chosen_ratio']:.2%}"
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
    parser.add_argument(
        "--traversal-cache-dir",
        default=None,
        help=(
            "Directory for reusable traversal caches. Each cache stores fixed "
            "queries plus Dijkstra and safe A* baseline results."
        ),
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
            traversal_cache_dir=args.traversal_cache_dir,
        )
        all_rows.extend(rows)

    write_csv(all_rows, Path(args.output))
    print()
    print(f"Saved ML-A* benchmark results to {args.output}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
