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

python "scripts/py/build_ml_dataset.py" "$@"
