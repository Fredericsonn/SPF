import argparse
import gzip
import json
import random
from collections import deque
from pathlib import Path


def parse_graph(path):
    graph = {}
    declared_nodes = None
    declared_edges = None

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("c"):
                continue

            parts = line.split()

            if parts[0] == "p":
                declared_nodes = int(parts[2])
                declared_edges = int(parts[3])
                for node in range(1, declared_nodes + 1):
                    graph[node] = {}

            elif parts[0] == "a":
                source = int(parts[1])
                target = int(parts[2])
                weight = int(parts[3])

                if source not in graph:
                    graph[source] = {}
                if target not in graph:
                    graph[target] = {}

                graph[source][target] = weight

    return graph, {
        "declared_nodes": declared_nodes,
        "declared_edges": declared_edges,
        "loaded_nodes": len(graph),
        "loaded_edges": sum(len(neighbors) for neighbors in graph.values()),
    }


def parse_coordinates(path):
    coords = {}
    declared_nodes = None

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("c"):
                continue

            parts = line.split()

            if parts[0] == "p":
                declared_nodes = int(parts[4])

            elif parts[0] == "v":
                node = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                coords[node] = [x, y]

    return coords, {
        "declared_nodes": declared_nodes,
        "loaded_coordinates": len(coords),
    }


def undirected_neighbors(graph):
    neighbors = {node: set(graph[node]) for node in graph}

    for source, outgoing in graph.items():
        for target in outgoing:
            if target not in neighbors:
                neighbors[target] = set()
            neighbors[target].add(source)

    return neighbors


def sample_connected_nodes(graph, node_count, rng, max_attempts=200):
    if node_count > len(graph):
        raise ValueError(
            f"Requested {node_count} nodes, but graph only has {len(graph)} nodes."
        )

    neighbors = undirected_neighbors(graph)
    valid_seeds = [node for node, values in neighbors.items() if values]

    for _ in range(max_attempts):
        seed = rng.choice(valid_seeds)
        selected = {seed}
        queue = deque([seed])

        while queue and len(selected) < node_count:
            current = queue.popleft()
            next_nodes = list(neighbors[current])
            rng.shuffle(next_nodes)

            for neighbor in next_nodes:
                if neighbor in selected:
                    continue

                selected.add(neighbor)
                queue.append(neighbor)

                if len(selected) >= node_count:
                    break

        if len(selected) == node_count:
            return seed, selected

    raise RuntimeError(
        f"Could not sample a connected subgraph with {node_count} nodes "
        f"after {max_attempts} attempts."
    )


def induced_edges(graph, selected_nodes):
    selected = set(selected_nodes)
    edges = []

    for source in selected:
        for target, weight in graph[source].items():
            if target in selected:
                edges.append([source, target, weight])

    return edges


def build_subgraph_payload(
    graph,
    coords,
    selected_nodes,
    edges,
    dataset_name,
    subgraph_name,
    seed_node,
    sampling_method,
):
    missing_coords = sorted(node for node in selected_nodes if node not in coords)

    if missing_coords:
        raise ValueError(
            f"{len(missing_coords)} selected nodes are missing coordinates. "
            f"First missing node: {missing_coords[0]}"
        )

    selected_sorted = sorted(selected_nodes)

    return {
        "name": subgraph_name,
        "source_dataset": dataset_name,
        "sampling_method": sampling_method,
        "seed_node": seed_node,
        "node_count": len(selected_sorted),
        "edge_count": len(edges),
        "nodes": selected_sorted,
        "coords": {str(node): coords[node] for node in selected_sorted},
        "edges": edges,
    }


def save_json_gz(payload, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wt", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate connected road-network subgraphs from DIMACS-style files."
    )
    parser.add_argument("--graph", required=True, help="Path to .gr/.txt edge file.")
    parser.add_argument("--coords", required=True, help="Path to .co/.txt coordinate file.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. paris or ny.")
    parser.add_argument("--output-dir", required=True, help="Directory for subgraph files.")
    parser.add_argument("--nodes", type=int, required=True, help="Wanted node count.")
    parser.add_argument("--count", type=int, default=1, help="Number of subgraphs to create.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    rng = random.Random(args.seed)

    graph, graph_meta = parse_graph(args.graph)
    coords, coord_meta = parse_coordinates(args.coords)

    print(f"Loaded graph: {graph_meta}")
    print(f"Loaded coordinates: {coord_meta}")

    output_dir = Path(args.output_dir)

    for index in range(1, args.count + 1):
        seed_node, selected_nodes = sample_connected_nodes(
            graph=graph,
            node_count=args.nodes,
            rng=rng,
        )

        edges = induced_edges(graph, selected_nodes)
        subgraph_name = f"{args.dataset}_{args.nodes}_{index:03d}"

        payload = build_subgraph_payload(
            graph=graph,
            coords=coords,
            selected_nodes=selected_nodes,
            edges=edges,
            dataset_name=args.dataset,
            subgraph_name=subgraph_name,
            seed_node=seed_node,
            sampling_method="randomized_bfs_connected_induced_subgraph",
        )

        output_path = output_dir / f"{subgraph_name}.json.gz"
        save_json_gz(payload, output_path)

        print(
            f"Saved {output_path} "
            f"({payload['node_count']} nodes, {payload['edge_count']} edges, "
            f"seed node {seed_node})"
        )


if __name__ == "__main__":
    main()
