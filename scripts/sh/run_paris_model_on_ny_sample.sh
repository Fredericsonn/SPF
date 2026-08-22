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

MODEL_PATH="${MODEL_PATH:-data/models/paris_all/hgb/paris_all_hgb.joblib}"
QUERIES="${QUERIES:-100}"
SEED="${SEED:-42}"
TRAVERSAL_CACHE_DIR="${TRAVERSAL_CACHE_DIR:-data/results/traversal_cache/paris_model_cross_city_ny}"
RESULT_ROOT="${RESULT_ROOT:-data/results/ml_astar/paris_model_cross_city_ny}"

run_benchmark() {
  local group="$1"
  local test_id="$2"
  local mode="$3"
  local scale="$4"
  local input_file="data/subgraphs/new_york/${group}_${test_id}.json.gz"
  local scale_dir="${scale//./_}"
  local output_file="${RESULT_ROOT}/${group}/${mode}/scale_${scale_dir}/test_${test_id}.csv"

  echo
  echo "============================================================"
  echo "Cross-city benchmark: group=${group}, test=${test_id}, mode=${mode}, scale=${scale}"
  echo "Model:     ${MODEL_PATH}"
  echo "Input:     ${input_file}"
  echo "Output:    ${output_file}"
  echo "Cache dir: ${TRAVERSAL_CACHE_DIR}"
  echo "============================================================"

  bash scripts/sh/benchmark_ml_astar.sh \
    --model "${MODEL_PATH}" \
    --input "${input_file}" \
    --output "${output_file}" \
    --queries "${QUERIES}" \
    --heuristic-mode "${mode}" \
    --ml-scale "${scale}" \
    --seed "${SEED}" \
    --traversal-cache-dir "${TRAVERSAL_CACHE_DIR}"
}

run_group() {
  local group="$1"
  local test_id="$2"

  run_benchmark "${group}" "${test_id}" "raw_ml" "1.00"
  run_benchmark "${group}" "${test_id}" "min_ml_geo" "1.00"
  run_benchmark "${group}" "${test_id}" "raw_ml" "0.25"
  run_benchmark "${group}" "${test_id}" "raw_ml" "0.50"
  run_benchmark "${group}" "${test_id}" "raw_ml" "0.75"
}

run_group "ny_1000" "009"

echo
echo "Paris model cross-city NY sample benchmarks complete."
