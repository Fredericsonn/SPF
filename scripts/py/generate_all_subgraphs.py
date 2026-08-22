import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PLAN = [
    {
        "dataset": "paris",
        "graph": "datasets/paris/paris_distance.txt",
        "coords": "datasets/paris/paris_coordinates.txt",
        "output_dir": "data/subgraphs/paris",
        "jobs": [
            {"nodes": 500, "count": 10},
            {"nodes": 1000, "count": 10},
            {"nodes": 2500, "count": 10},
            {"nodes": 5000, "count": 5},
        ],
    },
    {
        "dataset": "ny",
        "graph": "datasets/new york/USA-road-d.NY.gr",
        "coords": "datasets/new york/USA-road-d.NY.co",
        "output_dir": "data/subgraphs/new_york",
        "jobs": [
            {"nodes": 1000, "count": 10},
            {"nodes": 5000, "count": 10},
            {"nodes": 10000, "count": 10},
            {"nodes": 25000, "count": 5},
        ],
    },
]


def project_root():
    return Path(__file__).resolve().parents[2]


def run_generator(root, generator_path, dataset, graph, coords, output_dir, nodes, count, seed):
    command = [
        sys.executable,
        str(generator_path),
        "--graph",
        str(root / graph),
        "--coords",
        str(root / coords),
        "--dataset",
        dataset,
        "--output-dir",
        str(root / output_dir),
        "--nodes",
        str(nodes),
        "--count",
        str(count),
        "--seed",
        str(seed),
    ]

    print()
    print(
        f"Generating {count} {dataset} subgraph(s) "
        f"with {nodes} nodes each..."
    )
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the full planned set of Paris and New York subgraphs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for reproducible subgraph generation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate only tiny smoke-test subgraphs.",
    )

    args = parser.parse_args()
    root = project_root()
    generator_path = root / "scripts" / "py" / "subgraph_generator.py"

    if not generator_path.exists():
        raise FileNotFoundError(f"Missing generator script: {generator_path}")

    if args.quick:
        plan = [
            {
                "dataset": "paris",
                "graph": "datasets/paris/paris_distance.txt",
                "coords": "datasets/paris/paris_coordinates.txt",
                "output_dir": "data/subgraphs/paris",
                "jobs": [{"nodes": 50, "count": 1}],
            }
        ]
    else:
        plan = DEFAULT_PLAN

    for dataset_plan in plan:
        for index, job in enumerate(dataset_plan["jobs"], start=1):
            run_generator(
                root=root,
                generator_path=generator_path,
                dataset=dataset_plan["dataset"],
                graph=dataset_plan["graph"],
                coords=dataset_plan["coords"],
                output_dir=dataset_plan["output_dir"],
                nodes=job["nodes"],
                count=job["count"],
                seed=args.seed + index,
            )

    print()
    print("Subgraph generation complete.")


if __name__ == "__main__":
    main()
