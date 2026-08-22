#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

if [[ -d "sklearn-env" ]]; then
  # shellcheck disable=SC1091
  source "sklearn-env/Scripts/activate" 2>/dev/null || true
  # shellcheck disable=SC1091
  source "sklearn-env/bin/activate" 2>/dev/null || true
fi

TARGETS="${TARGETS:-50}"
SEED="${SEED:-42}"
MODEL_OUTPUT="${MODEL_OUTPUT:-data/models/ny_all/hgb/ny_all_hgb_T${TARGETS}.joblib}"
DATASET_SUFFIX="${DATASET_SUFFIX:-T${TARGETS}}"
REBUILD_DATASETS="${REBUILD_DATASETS:-0}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-}"

build_group() {
  local group="$1"
  local pattern="$2"
  local output_dir="data/ml_datasets/${group}_${DATASET_SUFFIX}"

  if [[ "${REBUILD_DATASETS}" != "1" && -d "${output_dir}" ]] && compgen -G "${output_dir}/*.npz" > /dev/null; then
    echo "Skipping ${group}: existing .npz files in ${output_dir}"
    return
  fi

  echo
  echo "Building ${group} ML datasets with ${TARGETS} targets per subgraph..."
  bash scripts/sh/build_ml_dataset.sh \
    --input data/subgraphs/new_york \
    --pattern "${pattern}" \
    --output-dir "${output_dir}" \
    --targets "${TARGETS}" \
    --seed "${SEED}"
}

build_group "ny_1000" "ny_1000_*.json.gz"
build_group "ny_5000" "ny_5000_*.json.gz"
build_group "ny_10000" "ny_10000_*.json.gz"
build_group "ny_25000" "ny_25000_*.json.gz"

TRAIN_ARGS=(
  --input-root data/ml_datasets
  --output "${MODEL_OUTPUT}"
  --model hist_gradient_boosting
  --groups "ny_1000_${DATASET_SUFFIX}" "ny_5000_${DATASET_SUFFIX}" "ny_10000_${DATASET_SUFFIX}" "ny_25000_${DATASET_SUFFIX}"
  --seed "${SEED}"
)

if [[ -n "${MAX_ROWS_PER_FILE}" ]]; then
  TRAIN_ARGS+=(--max-rows-per-file "${MAX_ROWS_PER_FILE}")
fi

echo
echo "Training NY-only HGB model..."
bash scripts/sh/train_mixed_heuristic_model.sh "${TRAIN_ARGS[@]}"
