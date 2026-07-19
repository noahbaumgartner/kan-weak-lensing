#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_base_name>" >&2
  echo "  Submits both KAT ConvStem Stage-2 sweeps (mse + score) via" >&2
  echo "  scripts/tune_kat.submit (runs on the plain 'gpu' partition, not" >&2
  echo "  gpu_top). experiment_base_name gets _mse / _score appended for the" >&2
  echo "  two MLflow experiments." >&2
  echo "" >&2
  echo "  Uses configs/sweep/image/tune_kat_model_mse.yaml and" >&2
  echo "  tune_kat_model_score.yaml, which already pin architecture/optimizer" >&2
  echo "  to their Stage-1 winners — no EXTRA_weak_lensing/OBJECTIVE needed." >&2
  exit 1
fi

BASE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting KAT Stage 2 (mse), experiment=${BASE}_mse..."
sbatch --job-name=kan_tune_kat_stage2_mse \
  --export=ALL,EXPERIMENT="${BASE}_mse",SWEEP=image/tune_kat_model_mse \
  "${SCRIPT_DIR}/tune_kat.submit"

echo "Submitting KAT Stage 2 (score), experiment=${BASE}_score..."
sbatch --job-name=kan_tune_kat_stage2_score \
  --export=ALL,EXPERIMENT="${BASE}_score",SWEEP=image/tune_kat_model_score \
  "${SCRIPT_DIR}/tune_kat.submit"

echo "Both KAT Stage 2 sweeps submitted. Check with: squeue -u \$USER"
