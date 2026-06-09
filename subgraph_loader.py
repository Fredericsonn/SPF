import argparse
import gzip
import json
from pathlib import Path


def open_json(path):
    path = Path(path)

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")

    return open(path, "r", encoding="utf-8")


def load_subgraph(path):
    with open_json(path) as file:
        payload = json.load(file)

    graph = {int(node): {} for node in payload["nodes"]}

    for source, target, weight in payload["edges"]:
        graph[int(source)][int(target)] = int(weight)

    coords = {
        int(node): (int(value[0]), int(value[1]))
        for node, value in payload["coords"].items()
    }

    metadata = {
        "name": payload.get("name"),
        "source_dataset": payload.get("source_dataset"),
        "sampling_method": payload.get("sampling_method"),
        "seed_node": payload.get("seed_node"),
        "node_count": payload.get("node_count", len(graph)),
        "edge_count": payload.get(
            "edge_count",
            sum(len(neighbors) for neighbors in graph.values()),
        ),
    }

    return graph, coords, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Load a generated subgraph JSON file and print a short summary."
    )
    parser.add_argument("path", help="Path to .json or .json.gz subgraph file.")

    args = parser.parse_args()
    graph, coords, metadata = load_subgraph(args.path)

    loaded_edges = sum(len(neighbors) for neighbors in graph.values())
    sample_node = next(iter(graph))

    print(f"Name: {metadata['name']}")
    print(f"Source dataset: {metadata['source_dataset']}")
    print(f"Nodes: {len(graph)}")
    print(f"Edges: {loaded_edges}")
    print(f"Sample node: {sample_node}")
    print(f"Sample neighbors: {graph[sample_node]}")
    print(f"Sample coordinates: {coords[sample_node]}")


if __name__ == "__main__":
    main()
