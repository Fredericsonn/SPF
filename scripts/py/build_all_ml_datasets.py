import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PLAN = [
    {
        "name": "paris_500",
        "input": "data/subgraphs/paris",
        "pattern": "paris_500_*.json.gz",
        "output_dir": "data/ml_datasets/paris_500",
        "targets": 50,
    },
    {
        "name": "paris_1000",
        "input": "data/subgraphs/paris",
        "pattern": "paris_1000_*.json.gz",
        "output_dir": "data/ml_datasets/paris_1000",
        "targets": 50,
    },
    {
        "name": "paris_2500",
        "input": "data/subgraphs/paris",
        "pattern": "paris_2500_*.json.gz",
        "output_dir": "data/ml_datasets/paris_2500",
        "targets": 50,
    },
    {
        "name": "paris_5000",
        "input": "data/subgraphs/paris",
        "pattern": "paris_5000_*.json.gz",
        "output_dir": "data/ml_datasets/paris_5000",
        "targets": 40,
    },
    {
        "name": "ny_1000",
        "input": "data/subgraphs/new_york",
        "pattern": "ny_1000_*.json.gz",
        "output_dir": "data/ml_datasets/ny_1000",
        "targets": 50,
    },
    {
        "name": "ny_5000",
        "input": "data/subgraphs/new_york",
        "pattern": "ny_5000_*.json.gz",
        "output_dir": "data/ml_datasets/ny_5000",
        "targets": 40,
    },
    {
        "name": "ny_10000",
        "input": "data/subgraphs/new_york",
        "pattern": "ny_10000_*.json.gz",
        "output_dir": "data/ml_datasets/ny_10000",
        "targets": 30,
    },
    {
        "name": "ny_25000",
        "input": "data/subgraphs/new_york",
        "pattern": "ny_25000_*.json.gz",
        "output_dir": "data/ml_datasets/ny_25000",
        "targets": 20,
    },
]


def project_root():
    return Path(__file__).resolve().parents[2]


def run_builder(root, builder_path, item, seed, overwrite):
    output_dir = root / item["output_dir"]

    if output_dir.exists() and list(output_dir.glob("*.npz")) and not overwrite:
        print(f"Skipping {item['name']}: existing .npz files in {output_dir}")
        return

    command = [
        sys.executable,
        str(builder_path),
        "--input",
        str(root / item["input"]),
        "--pattern",
        item["pattern"],
        "--output-dir",
        str(output_dir),
        "--targets",
        str(item["targets"]),
        "--seed",
        str(seed),
    ]

    if item.get("max_nodes_per_target") is not None:
        command.extend(["--max-nodes-per-target", str(item["max_nodes_per_target"])])

    print()
    print(
        f"Building ML datasets for {item['name']} "
        f"({item['targets']} targets per subgraph)..."
    )
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build ML datasets for all planned Paris and New York subgraphs."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild groups even when .npz files already exist.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional group names to build, e.g. paris_500 ny_1000.",
    )

    args = parser.parse_args()
    root = project_root()
    builder_path = root / "scripts" / "py" / "build_ml_dataset.py"

    plan = DEFAULT_PLAN
    if args.only:
        requested = set(args.only)
        plan = [item for item in plan if item["name"] in requested]
        missing = requested - {item["name"] for item in plan}
        if missing:
            raise ValueError(f"Unknown group(s): {sorted(missing)}")

    for index, item in enumerate(plan, start=1):
        run_builder(
            root=root,
            builder_path=builder_path,
            item=item,
            seed=args.seed + index,
            overwrite=args.overwrite,
        )

    print()
    print("All requested ML datasets are ready.")


if __name__ == "__main__":
    main()
