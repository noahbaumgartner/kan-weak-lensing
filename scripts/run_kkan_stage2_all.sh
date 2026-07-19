#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_base_name>" >&2
  echo "  Submits both KKAN ConvStem Stage-2 sweeps (mse + score) via" >&2
  echo "  scripts/tune_kkan.submit, forced onto the 'gpu_top' partition" >&2
  echo "  (overrides that script's own --partition=gpu via sbatch CLI flag," >&2
  echo "  so other kkan submissions using tune_kkan.submit are unaffected)." >&2
  echo "  experiment_base_name gets _mse / _score appended for the two" >&2
  echo "  MLflow experiments." >&2
  echo "" >&2
  echo "  Uses configs/sweep/image/tune_kkan_model_mse.yaml and" >&2
  echo "  tune_kkan_model_score.yaml, which already pin optimizer/batch_size" >&2
  echo "  to their Stage-1 winners — no EXTRA_weak_lensing/OBJECTIVE needed." >&2
  exit 1
fi

BASE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting KKAN Stage 2 (mse), experiment=${BASE}_mse, partition=gpu_top..."
sbatch --job-name=kan_tune_kkan_stage2_mse --partition=gpu_top \
  --export=ALL,EXPERIMENT="${BASE}_mse",SWEEP=image/tune_kkan_model_mse \
  "${SCRIPT_DIR}/tune_kkan.submit"

echo "Submitting KKAN Stage 2 (score), experiment=${BASE}_score, partition=gpu_top..."
sbatch --job-name=kan_tune_kkan_stage2_score --partition=gpu_top \
  --export=ALL,EXPERIMENT="${BASE}_score",SWEEP=image/tune_kkan_model_score \
  "${SCRIPT_DIR}/tune_kkan.submit"

echo "Both KKAN Stage 2 sweeps submitted. Check with: squeue -u \$USER"
