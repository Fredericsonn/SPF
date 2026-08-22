import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_npz_dataset(path):
    data = np.load(path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    metadata = json.loads(str(data["metadata"]))
    return X, y, metadata


def collect_dataset_files(input_path, pattern):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched {input_path / pattern}")

    return files


def file_index(path):
    stem = path.stem
    parts = stem.split("_")

    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    return None


def split_files(files, train_max_index, validation_index):
    train_files = []
    validation_files = []

    for path in files:
        index = file_index(path)

        if index is None:
            continue

        if index <= train_max_index:
            train_files.append(path)
        elif index == validation_index:
            validation_files.append(path)

    if not train_files:
        raise RuntimeError("No training files selected.")

    if not validation_files:
        raise RuntimeError("No validation files selected.")

    return train_files, validation_files


def load_many(files):
    X_parts = []
    y_parts = []
    metadata = []

    for path in files:
        X, y, item_metadata = load_npz_dataset(path)
        X_parts.append(X)
        y_parts.append(y)
        metadata.append(item_metadata)
        print(f"Loaded {path}: X={X.shape}, y={y.shape}")

    return np.vstack(X_parts), np.concatenate(y_parts), metadata


def make_model(model_type, random_state, n_jobs):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model, X, y, label):
    predictions = model.predict(X)
    errors = predictions - y
    abs_errors = np.abs(errors)
    overestimation_rate = float(np.mean(predictions > y))

    metrics = {
        f"{label}_rows": int(X.shape[0]),
        f"{label}_mae": float(mean_absolute_error(y, predictions)),
        f"{label}_rmse": float(mean_squared_error(y, predictions) ** 0.5),
        f"{label}_r2": float(r2_score(y, predictions)),
        f"{label}_mean_true_distance": float(np.mean(y)),
        f"{label}_mean_prediction": float(np.mean(predictions)),
        f"{label}_median_abs_error": float(np.median(abs_errors)),
        f"{label}_p95_abs_error": float(np.percentile(abs_errors, 95)),
        f"{label}_overestimation_rate": overestimation_rate,
        f"{label}_mean_overestimation": float(
            np.mean(np.maximum(predictions - y, 0))
        ),
    }

    print()
    print(f"{label.upper()} METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def save_model_bundle(model, output_path, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "metadata": metadata,
        },
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a supervised model for learned A* heuristic estimation."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing .npz ML datasets, or one .npz file.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern when --input is a directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .joblib model path.",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "hist_gradient_boosting"],
        default="hist_gradient_boosting",
        help="Regression model type.",
    )
    parser.add_argument(
        "--train-max-index",
        type=int,
        default=7,
        help="Use files with suffix index <= this value for training.",
    )
    parser.add_argument(
        "--validation-index",
        type=int,
        default=8,
        help="Use files with this suffix index for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for supported models.",
    )

    args = parser.parse_args()
    files = collect_dataset_files(args.input, args.pattern)
    train_files, validation_files = split_files(
        files,
        train_max_index=args.train_max_index,
        validation_index=args.validation_index,
    )

    print("Training files:")
    for path in train_files:
        print(f"  {path}")

    print("Validation files:")
    for path in validation_files:
        print(f"  {path}")

    X_train, y_train, train_metadata = load_many(train_files)
    X_validation, y_validation, validation_metadata = load_many(validation_files)

    print()
    print(f"Training matrix: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation matrix: X={X_validation.shape}, y={y_validation.shape}")

    model = make_model(
        model_type=args.model,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    print()
    print(f"Training model: {args.model}")
    model.fit(X_train, y_train)

    train_metrics = evaluate(model, X_train, y_train, "train")
    validation_metrics = evaluate(model, X_validation, y_validation, "validation")

    output_metadata = {
        "model_type": args.model,
        "input": args.input,
        "pattern": args.pattern,
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
        "train_source_metadata": train_metadata,
        "validation_source_metadata": validation_metadata,
        "metrics": {
            **train_metrics,
            **validation_metrics,
        },
        "seed": args.seed,
    }

    save_model_bundle(model, args.output, output_metadata)
    print()
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
