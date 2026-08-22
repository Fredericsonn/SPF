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

SEED="${SEED:-42}"
TARGET_COUNTS="${TARGET_COUNTS:-250 500 1000 2500}"
REBUILD_DATASETS="${REBUILD_DATASETS:-0}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-}"

mkdir -p data/results/logs

eligible_groups_for_targets() {
  local targets="$1"

  if (( targets <= 500 )); then
    echo "paris_500 paris_1000 paris_2500 paris_5000"
  elif (( targets <= 1000 )); then
    echo "paris_1000 paris_2500 paris_5000"
  elif (( targets <= 2500 )); then
    echo "paris_2500 paris_5000"
  elif (( targets <= 5000 )); then
    echo "paris_5000"
  else
    echo ""
  fi
}

pattern_for_group() {
  local group="$1"
  echo "${group}_*.json.gz"
}

build_group() {
  local group="$1"
  local targets="$2"
  local suffix="T${targets}"
  local output_dir="data/ml_datasets/${group}_${suffix}"

  if [[ "${REBUILD_DATASETS}" != "1" && -d "${output_dir}" ]] && compgen -G "${output_dir}/*.npz" > /dev/null; then
    echo "Skipping ${group} T${targets}: existing .npz files in ${output_dir}"
    return
  fi

  echo
  echo "Building ${group} datasets with ${targets} targets per subgraph..."
  bash scripts/sh/build_ml_dataset.sh \
    --input data/subgraphs/paris \
    --pattern "$(pattern_for_group "${group}")" \
    --output-dir "${output_dir}" \
    --targets "${targets}" \
    --seed "${SEED}"
}

train_for_targets() {
  local targets="$1"
  local suffix="T${targets}"
  local groups
  local suffixed_groups=()

  groups="$(eligible_groups_for_targets "${targets}")"

  if [[ -z "${groups}" ]]; then
    echo "No eligible Paris groups for T${targets}; skipping."
    return
  fi

  echo
  echo "============================================================"
  echo "Paris target sweep: T${targets}"
  echo "Eligible groups: ${groups}"
  echo "============================================================"

  for group in ${groups}; do
    build_group "${group}" "${targets}"
    suffixed_groups+=("${group}_${suffix}")
  done

  local output_path="data/models/paris_target_sweep/hgb/paris_hgb_${suffix}.joblib"
  local log_path="data/results/logs/train_paris_hgb_${suffix}.log"
  local train_args=(
    --input-root data/ml_datasets
    --output "${output_path}"
    --model hist_gradient_boosting
    --groups "${suffixed_groups[@]}"
    --seed "${SEED}"
    --verbose 1
  )

  if [[ -n "${MAX_ROWS_PER_FILE}" ]]; then
    train_args+=(--max-rows-per-file "${MAX_ROWS_PER_FILE}")
  fi

  echo
  echo "Training Paris HGB model for T${targets}..."
  echo "Model output: ${output_path}"
  echo "Log output:   ${log_path}"
  bash scripts/sh/train_mixed_heuristic_model.sh "${train_args[@]}" \
    | tee "${log_path}"
}

for targets in ${TARGET_COUNTS}; do
  train_for_targets "${targets}"
done

echo
echo "Paris target-count sweep complete."
