#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per MLP-style KAN model" >&2
  echo "  (fastkan, fasterkan, efficientkan, wavkan) on the weak_lensing dataset," >&2
  echo "  each sweeping over the dimension-reduction method" >&2
  echo "  (avgpool / kymatio). The experiment_name is used as the MLflow" >&2
  echo "  experiment for all runs." >&2
  exit 1
fi

EXPERIMENT="$1"
MODELS=(fastkan fasterkan efficientkan wavkan)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in "${MODELS[@]}"; do
  job="${SCRIPT_DIR}/tune_${model}_wl.submit"
  echo "Submitting ${job} (experiment=${EXPERIMENT})..."
  sbatch --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#MODELS[@]} weak-lensing tuning jobs submitted. Check with: squeue -u \$USER"
