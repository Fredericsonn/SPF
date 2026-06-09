import osmnx as ox
import json

G = ox.graph_from_place(
    "Paris, France",
    network_type="drive"
)

G = ox.convert.to_digraph(
    G,
    weight="length"
)

nodes = list(G.nodes())
edges = list(G.edges(data=True))

node_to_idx = {
    node: i + 1
    for i, node in enumerate(nodes)
}

with open("datasets/paris_coordinates.txt", "w", encoding="utf-8") as f:

    f.write("c Paris road network coordinates\n")
    f.write(f"p aux sp co {len(nodes)}\n")

    for node in nodes:
        data = G.nodes[node]

        x = int(data["x"] * 1_000_000)
        y = int(data["y"] * 1_000_000)

        f.write(
            f"v {node_to_idx[node]} {x} {y}\n"
        )


with open("datasets/paris_distance.txt", "w", encoding="utf-8") as f:

    f.write("c Paris road network graph\n")
    f.write(
        f"p sp {len(nodes)} {len(edges)}\n"
    )

    for u, v, data in edges:

        weight = int(
            data.get("length", 1)
        )

        f.write(
            f"a {node_to_idx[u]} "
            f"{node_to_idx[v]} "
            f"{weight}\n"
        )


with open(
    "datasets/paris_id_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {str(k): v for k, v in node_to_idx.items()},
        f,
        indent=2
    )

print("Export finished")