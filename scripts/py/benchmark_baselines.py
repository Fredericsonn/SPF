import argparse
import csv
import heapq
import math
import random
import time
from pathlib import Path

from subgraph_loader import load_subgraph


CSV_FIELDS = [
    "subgraph",
    "source_dataset",
    "node_count",
    "edge_count",
    "query_index",
    "source",
    "target",
    "dijkstra_cost",
    "dijkstra_expanded",
    "dijkstra_runtime_ms",
    "astar_cost",
    "astar_expanded",
    "astar_runtime_ms",
    "astar_cost_ratio",
    "astar_expanded_ratio",
]


def euclidean(coords, node, target):
    lon1_raw, lat1_raw = coords[node]
    lon2_raw, lat2_raw = coords[target]

    lon1 = math.radians(lon1_raw / 1_000_000)
    lat1 = math.radians(lat1_raw / 1_000_000)
    lon2 = math.radians(lon2_raw / 1_000_000)
    lat2 = math.radians(lat2_raw / 1_000_000)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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

        if current_distance > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return math.inf, expanded


def astar_point_to_point(graph, coords, source, target):
    g_score = {node: math.inf for node in graph}
    g_score[source] = 0.0
    queue = [(euclidean(coords, source, target), source)]
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
                priority = tentative_g + euclidean(coords, neighbor, target)
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


def benchmark_subgraph(subgraph_path, query_count, seed, max_attempts):
    graph, coords, metadata = load_subgraph(subgraph_path)
    rng = random.Random(seed)
    queries = sample_reachable_queries(graph, query_count, rng, max_attempts)
    rows = []

    print(
        f"Benchmarking {metadata['name']} "
        f"({metadata['node_count']} nodes, {metadata['edge_count']} edges, "
        f"{query_count} queries)..."
    )

    for query_index, (source, target) in enumerate(queries, start=1):
        dijkstra_cost, dijkstra_expanded, dijkstra_runtime_ms = timed_call(
            dijkstra_point_to_point,
            graph,
            source,
            target,
        )
        astar_cost, astar_expanded, astar_runtime_ms = timed_call(
            astar_point_to_point,
            graph,
            coords,
            source,
            target,
        )

        if dijkstra_cost == 0 or math.isinf(dijkstra_cost):
            astar_cost_ratio = math.inf
        else:
            astar_cost_ratio = astar_cost / dijkstra_cost

        if dijkstra_expanded == 0:
            astar_expanded_ratio = math.inf
        else:
            astar_expanded_ratio = astar_expanded / dijkstra_expanded

        rows.append(
            {
                "subgraph": metadata["name"],
                "source_dataset": metadata["source_dataset"],
                "node_count": metadata["node_count"],
                "edge_count": metadata["edge_count"],
                "query_index": query_index,
                "source": source,
                "target": target,
                "dijkstra_cost": dijkstra_cost,
                "dijkstra_expanded": dijkstra_expanded,
                "dijkstra_runtime_ms": dijkstra_runtime_ms,
                "astar_cost": astar_cost,
                "astar_expanded": astar_expanded,
                "astar_runtime_ms": astar_runtime_ms,
                "astar_cost_ratio": astar_cost_ratio,
                "astar_expanded_ratio": astar_expanded_ratio,
            }
        )

        print(
            f"  query {query_index}/{query_count}: "
            f"{source}->{target}, "
            f"Dijkstra expanded={dijkstra_expanded}, "
            f"A* expanded={astar_expanded}"
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
        description="Benchmark Dijkstra and Euclidean A* on generated subgraphs."
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
        "--output",
        required=True,
        help="CSV file where benchmark results will be saved.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Reachable source-target queries sampled per subgraph.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Maximum attempts to sample reachable queries per subgraph.",
    )

    args = parser.parse_args()
    input_files = collect_input_files(args.input, args.pattern)
    all_rows = []

    for index, subgraph_path in enumerate(input_files, start=1):
        rows = benchmark_subgraph(
            subgraph_path=subgraph_path,
            query_count=args.queries,
            seed=args.seed + index,
            max_attempts=args.max_attempts,
        )
        all_rows.extend(rows)

    write_csv(all_rows, Path(args.output))
    print()
    print(f"Saved baseline benchmark results to {args.output}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
