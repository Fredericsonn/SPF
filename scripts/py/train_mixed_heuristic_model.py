import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GROUP_PLAN = {
    "paris_500": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "paris_1000": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "paris_2500": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "paris_5000": {"train": range(1, 4), "validation": {4}, "test": {5}},
    "ny_1000": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "ny_5000": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "ny_10000": {"train": range(1, 8), "validation": {8}, "test": {9, 10}},
    "ny_25000": {"train": range(1, 4), "validation": {4}, "test": {5}},
}


def base_group_name(group):
    for known_group in sorted(GROUP_PLAN, key=len, reverse=True):
        if group == known_group or group.startswith(f"{known_group}_"):
            return known_group

    return group


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def file_index(path):
    parts = path.stem.split("_")

    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    return None


def collect_files(input_root, groups):
    input_root = Path(input_root)
    train_files = []
    validation_files = []
    ignored_test_files = []

    for group in groups:
        group_dir = input_root / group

        if not group_dir.exists():
            raise FileNotFoundError(f"Missing dataset group directory: {group_dir}")

        base_group = base_group_name(group)
        plan = GROUP_PLAN[base_group]

        for path in sorted(group_dir.glob("*.npz")):
            index = file_index(path)

            if index in plan["train"]:
                train_files.append(path)
            elif index in plan["validation"]:
                validation_files.append(path)
            elif index in plan["test"]:
                ignored_test_files.append(path)

    if not train_files:
        raise RuntimeError("No training files selected.")

    if not validation_files:
        raise RuntimeError("No validation files selected.")

    return train_files, validation_files, ignored_test_files


def load_npz_dataset(path):
    data = np.load(path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    metadata = json.loads(str(data["metadata"]))
    return X, y, metadata


def load_many(files, max_rows_per_file, seed):
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    metadata = []

    for path in files:
        started = time.perf_counter()
        X, y, item_metadata = load_npz_dataset(path)

        if max_rows_per_file is not None and X.shape[0] > max_rows_per_file:
            indices = rng.choice(X.shape[0], size=max_rows_per_file, replace=False)
            X = X[indices]
            y = y[indices]
            item_metadata = {
                **item_metadata,
                "sampled_rows_for_training": int(max_rows_per_file),
            }

        X_parts.append(X)
        y_parts.append(y)
        metadata.append(item_metadata)
        elapsed = time.perf_counter() - started
        log(f"Loaded {path}: X={X.shape}, y={y.shape} in {elapsed:.2f}s")

    log("Stacking loaded arrays...")
    started = time.perf_counter()
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    elapsed = time.perf_counter() - started
    log(f"Stacked arrays: X={X.shape}, y={y.shape} in {elapsed:.2f}s")

    return X, y, metadata


def make_model(model_type, random_state, n_jobs, verbose):
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
            verbose=verbose,
        )

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=240,
            max_depth=28,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model, X, y, label):
    log(f"Predicting {label} set...")
    started = time.perf_counter()
    predictions = model.predict(X)
    prediction_elapsed = time.perf_counter() - started

    log(f"Computing {label} metrics...")
    errors = predictions - y
    abs_errors = np.abs(errors)

    metrics = {
        f"{label}_rows": int(X.shape[0]),
        f"{label}_mae": float(mean_absolute_error(y, predictions)),
        f"{label}_rmse": float(mean_squared_error(y, predictions) ** 0.5),
        f"{label}_r2": float(r2_score(y, predictions)),
        f"{label}_mean_true_distance": float(np.mean(y)),
        f"{label}_mean_prediction": float(np.mean(predictions)),
        f"{label}_median_abs_error": float(np.median(abs_errors)),
        f"{label}_p95_abs_error": float(np.percentile(abs_errors, 95)),
        f"{label}_overestimation_rate": float(np.mean(predictions > y)),
        f"{label}_mean_overestimation": float(
            np.mean(np.maximum(predictions - y, 0))
        ),
    }

    print()
    print(f"{label.upper()} METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"{label}_prediction_time_s: {prediction_elapsed}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a mixed learned-heuristic model from multiple dataset groups."
    )
    parser.add_argument("--input-root", default="data/ml_datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=list(GROUP_PLAN),
        help="Dataset groups to include.",
    )
    parser.add_argument(
        "--model",
        choices=["hist_gradient_boosting", "random_forest"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Model verbosity. HistGradientBoosting prints iteration progress when > 0.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row cap per .npz file for faster/lighter training.",
    )

    args = parser.parse_args()

    unknown = {group for group in args.groups if base_group_name(group) not in GROUP_PLAN}
    if unknown:
        raise ValueError(f"Unknown group(s): {sorted(unknown)}")

    train_files, validation_files, test_files = collect_files(
        input_root=args.input_root,
        groups=args.groups,
    )

    log("Training files:")
    for path in train_files:
        print(f"  {path}")

    log("Validation files:")
    for path in validation_files:
        print(f"  {path}")

    log("Held-out test files not used for training:")
    for path in test_files:
        print(f"  {path}")

    log("Loading training datasets...")
    X_train, y_train, train_metadata = load_many(
        train_files,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed,
    )
    log("Loading validation datasets...")
    X_validation, y_validation, validation_metadata = load_many(
        validation_files,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed + 1,
    )

    print()
    log(f"Training matrix: X={X_train.shape}, y={y_train.shape}")
    log(f"Validation matrix: X={X_validation.shape}, y={y_validation.shape}")

    model = make_model(
        model_type=args.model,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    print()
    log(f"Training mixed model: {args.model}")
    training_started = time.perf_counter()
    model.fit(X_train, y_train)
    training_elapsed = time.perf_counter() - training_started
    log(f"Finished model training in {training_elapsed:.2f}s")

    train_metrics = evaluate(model, X_train, y_train, "train")
    validation_metrics = evaluate(model, X_validation, y_validation, "validation")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_type": args.model,
        "groups": args.groups,
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
        "held_out_test_files": [str(path) for path in test_files],
        "train_source_metadata": train_metadata,
        "validation_source_metadata": validation_metadata,
        "max_rows_per_file": args.max_rows_per_file,
        "metrics": {
            **train_metrics,
            **validation_metrics,
        },
        "seed": args.seed,
    }

    log(f"Saving model to {output_path}...")
    joblib.dump({"model": model, "metadata": metadata}, output_path)
    print()
    log(f"Saved mixed model to {output_path}")


if __name__ == "__main__":
    main()
