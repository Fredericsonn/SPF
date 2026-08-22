import argparse
import csv
import json
import math
import random
from pathlib import Path

import joblib

from benchmark_ml_astar import (
    CSV_FIELDS,
    compute_safe_geo_scale,
    dijkstra_point_to_point,
    make_heuristic_function,
    ml_astar_point_to_point,
    safe_geo_astar_point_to_point,
    timed_call,
)
from subgraph_generator import (
    build_subgraph_payload,
    induced_edges,
    parse_coordinates,
    parse_graph,
    sample_connected_nodes,
)

def equirectangular_distance_m(coord_a, coord_b):
    lon1_raw, lat1_raw = coord_a
    lon2_raw, lat2_raw = coord_b

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


def generate_synthetic_graph(
    node_count,
    neighbors,
    seed,
    center_lon,
    center_lat,
    spread_m,
    min_road_factor,
    max_road_factor,
):
    if node_count < 2:
        raise ValueError("--nodes must be at least 2")
    if neighbors < 1:
        raise ValueError("--neighbors must be at least 1")

    rng = random.Random(seed)
    coords = {}
    meters_per_lat_degree = 111_320
    meters_per_lon_degree = 111_320 * math.cos(math.radians(center_lat))

    for node in range(1, node_count + 1):
        dx_m = rng.uniform(-spread_m, spread_m)
        dy_m = rng.uniform(-spread_m, spread_m)
        lon = center_lon + dx_m / meters_per_lon_degree
        lat = center_lat + dy_m / meters_per_lat_degree
        coords[node] = (int(round(lon * 1_000_000)), int(round(lat * 1_000_000)))

    graph = {node: {} for node in coords}

    # Ring edges guarantee weak connectivity before the nearest-neighbor links are added.
    ordered = list(coords)
    for index, source in enumerate(ordered):
        target = ordered[(index + 1) % node_count]
        distance = equirectangular_distance_m(coords[source], coords[target])
        factor = rng.uniform(min_road_factor, max_road_factor)
        weight = max(1, int(round(distance * factor)))
        graph[source][target] = min(graph[source].get(target, weight), weight)
        graph[target][source] = min(graph[target].get(source, weight), weight)

    effective_neighbors = min(neighbors, node_count - 1)
    for source in ordered:
        nearest = sorted(
            (
                (equirectangular_distance_m(coords[source], coords[target]), target)
                for target in ordered
                if target != source
            ),
            key=lambda item: item[0],
        )[:effective_neighbors]

        for distance, target in nearest:
            factor = rng.uniform(min_road_factor, max_road_factor)
            weight = max(1, int(round(distance * factor)))
            graph[source][target] = min(graph[source].get(target, weight), weight)
            graph[target][source] = min(graph[target].get(source, weight), weight)

    edge_count = sum(len(neighbors_map) for neighbors_map in graph.values())
    return graph, coords, edge_count


def sample_reachable_queries(graph, query_count, seed, max_attempts):
    rng = random.Random(seed)
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


def benchmark_generated_graph(
    model,
    graph,
    coords,
    metadata,
    query_count,
    seed,
    max_attempts,
    heuristic_mode,
    ml_scale,
):
    geo_scale = compute_safe_geo_scale(graph, coords)
    heuristic, tracker = make_heuristic_function(
        model=model,
        graph=graph,
        coords=coords,
        heuristic_mode=heuristic_mode,
        ml_scale=ml_scale,
        geo_scale=geo_scale,
    )
    queries = sample_reachable_queries(graph, query_count, seed, max_attempts)
    rows = []

    print(
        f"Benchmarking synthetic graph {metadata['name']} "
        f"({metadata['node_count']} nodes, {metadata['edge_count']} edges, "
        f"{query_count} queries, mode={heuristic_mode}, "
        f"ml_scale={ml_scale}, geo_scale={geo_scale:.6f})...",
        flush=True,
    )

    for query_index, (source, target) in enumerate(queries, start=1):
        dijkstra_cost, dijkstra_expanded, dijkstra_runtime_ms = timed_call(
            dijkstra_point_to_point,
            graph,
            source,
            target,
        )
        safe_cost, safe_expanded, safe_runtime_ms = timed_call(
            safe_geo_astar_point_to_point,
            graph,
            coords,
            source,
            target,
            geo_scale,
        )

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
            }
        )

        print(
            f"  query {query_index}/{query_count}: "
            f"{source}->{target}, "
            f"Dijkstra expanded={dijkstra_expanded}, "
            f"safe A* expanded={safe_expanded}, "
            f"ML-A* expanded={ml_expanded}, "
            f"cost_ratio={cost_ratio:.6f}, "
            f"ml_chosen={heuristic_stats['ml_chosen_ratio']:.2%}, "
            f"geo_chosen={heuristic_stats['geo_chosen_ratio']:.2%}",
            flush=True,
        )

    return rows


def summarize(rows):
    query_count = len(rows)
    optimal_count = sum(1 for row in rows if row["optimal"])

    def mean(field):
        return sum(row[field] for row in rows) / query_count if query_count else 0.0

    return {
        "queries": query_count,
        "optimal_rate": optimal_count / query_count if query_count else 0.0,
        "bad_rows": query_count - optimal_count,
        "mean_cost_ratio": mean("ml_astar_cost_ratio"),
        "max_cost_ratio": max((row["ml_astar_cost_ratio"] for row in rows), default=0.0),
        "mean_expanded_ratio": mean("ml_astar_expanded_ratio"),
        "mean_dijkstra_runtime_ms": mean("dijkstra_runtime_ms"),
        "mean_safe_astar_runtime_ms": mean("safe_astar_runtime_ms"),
        "mean_ml_astar_runtime_ms": mean("ml_astar_runtime_ms"),
        "mean_dijkstra_expanded": mean("dijkstra_expanded"),
        "mean_safe_astar_expanded": mean("safe_astar_expanded"),
        "mean_ml_astar_expanded": mean("ml_astar_expanded"),
        "mean_ml_chosen_ratio": mean("ml_chosen_ratio"),
        "mean_geo_chosen_ratio": mean("geo_chosen_ratio"),
    }


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        "safe_astar_cost",
        "safe_astar_expanded",
        "safe_astar_runtime_ms",
        "safe_astar_cost_ratio",
        "safe_astar_expanded_ratio",
    ]
    fields = CSV_FIELDS + extra_fields

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_subgraph(graph_path, coords_path, dataset_name, node_count, seed):
    rng = random.Random(seed)
    full_graph, graph_meta = parse_graph(graph_path)
    full_coords, coord_meta = parse_coordinates(coords_path)

    print(f"Loaded source graph: {graph_meta}", flush=True)
    print(f"Loaded source coordinates: {coord_meta}", flush=True)

    seed_node, selected_nodes = sample_connected_nodes(
        graph=full_graph,
        node_count=node_count,
        rng=rng,
    )
    edges = induced_edges(full_graph, selected_nodes)
    subgraph_name = f"{dataset_name}_runtime_{node_count}_seed{seed}"
    payload = build_subgraph_payload(
        graph=full_graph,
        coords=full_coords,
        selected_nodes=selected_nodes,
        edges=edges,
        dataset_name=dataset_name,
        subgraph_name=subgraph_name,
        seed_node=seed_node,
        sampling_method="runtime_randomized_bfs_connected_induced_subgraph",
    )

    graph = {node: {} for node in payload["nodes"]}
    for source, target, weight in payload["edges"]:
        graph[source][target] = weight

    coords = {int(node): tuple(values) for node, values in payload["coords"].items()}
    metadata = {
        "name": payload["name"],
        "source_dataset": payload["source_dataset"],
        "node_count": payload["node_count"],
        "edge_count": payload["edge_count"],
        "seed_node": payload["seed_node"],
        "sampling_method": payload["sampling_method"],
    }
    return graph, coords, metadata


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a graph at runtime, then compare Dijkstra, safe geographic "
            "A*, and ML-A*. The graph can be fully synthetic or sampled from an "
            "existing DIMACS-style dataset."
        )
    )
    parser.add_argument("--model", required=True, help="Path to .joblib model bundle.")
    parser.add_argument("--output", required=True, help="Output query-level CSV path.")
    parser.add_argument("--nodes", type=int, required=True, help="Generated node count.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "dataset"],
        default="synthetic",
        help="Generate a random geometric graph or sample a subgraph from raw data.",
    )
    parser.add_argument("--graph", help="Dataset mode: path to .gr/.txt edge file.")
    parser.add_argument("--coords", help="Dataset mode: path to .co/.txt coordinate file.")
    parser.add_argument("--dataset", default="runtime", help="Dataset mode: dataset name.")
    parser.add_argument("--neighbors", type=int, default=4, help="Synthetic mode: nearest neighbors per node.")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument(
        "--heuristic-mode",
        choices=["raw_ml", "min_ml_geo", "max_ml_geo", "geo_only"],
        default="raw_ml",
    )
    parser.add_argument("--ml-scale", type=float, default=0.25)
    parser.add_argument("--center-lon", type=float, default=-73.9857)
    parser.add_argument("--center-lat", type=float, default=40.7484)
    parser.add_argument("--spread-m", type=float, default=10000.0)
    parser.add_argument("--min-road-factor", type=float, default=1.05)
    parser.add_argument("--max-road-factor", type=float, default=1.60)

    args = parser.parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    if isinstance(bundle, dict) and "metadata" in bundle:
        print("Loaded model metadata:")
        print(json.dumps(bundle["metadata"].get("metrics", {}), indent=2))

    if args.source == "dataset":
        if not args.graph or not args.coords:
            raise ValueError("Dataset mode requires both --graph and --coords.")
        graph, coords, metadata = load_dataset_subgraph(
            graph_path=args.graph,
            coords_path=args.coords,
            dataset_name=args.dataset,
            node_count=args.nodes,
            seed=args.seed,
        )
    else:
        graph, coords, edge_count = generate_synthetic_graph(
            node_count=args.nodes,
            neighbors=args.neighbors,
            seed=args.seed,
            center_lon=args.center_lon,
            center_lat=args.center_lat,
            spread_m=args.spread_m,
            min_road_factor=args.min_road_factor,
            max_road_factor=args.max_road_factor,
        )
        metadata = {
            "name": f"synthetic_{args.nodes}_k{args.neighbors}_seed{args.seed}",
            "source_dataset": "synthetic_random_geometric",
            "node_count": args.nodes,
            "edge_count": edge_count,
        }

    rows = benchmark_generated_graph(
        model=model,
        graph=graph,
        coords=coords,
        metadata=metadata,
        query_count=args.queries,
        seed=args.seed + 1,
        max_attempts=args.max_attempts,
        heuristic_mode=args.heuristic_mode,
        ml_scale=args.ml_scale,
    )
    output_path = Path(args.output)
    write_csv(rows, output_path)

    summary = summarize(rows)
    summary.update(
        {
            "graph_source": args.source,
            "generated_graph": metadata,
            "heuristic_mode": args.heuristic_mode,
            "ml_scale": args.ml_scale,
            "seed": args.seed,
        }
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Saved generated-graph benchmark results to {output_path}")
    print(f"Saved generated-graph benchmark summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
