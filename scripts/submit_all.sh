#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per KAN model on the weak_lensing dataset:" >&2
  echo "  MLP-style (fastkan, fasterkan, efficientkan, wavkan) + image models" >&2
  echo "  (kkan, kat). The MLP-style jobs sweep the dimension-reduction method" >&2
  echo "  (avgpool / conv); kkan/kat consume the image directly. The" >&2
  echo "  experiment_name is used as the MLflow experiment for all runs." >&2
  echo "  Note: kkan/kat are not yet adapted to weak lensing and may fail" >&2
  echo "  (see README 'Hinweis zu kkan / kat')." >&2
  exit 1
fi

EXPERIMENT="$1"
# One submit file per model.
JOBS=(
  tune_fastkan.submit
  tune_fasterkan.submit
  tune_efficientkan.submit
  tune_wavkan.submit
  tune_kkan.submit
  tune_kat.submit
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for job_name in "${JOBS[@]}"; do
  job="${SCRIPT_DIR}/${job_name}"
  echo "Submitting ${job} (experiment=${EXPERIMENT})..."
  sbatch --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#JOBS[@]} weak-lensing tuning jobs submitted. Check with: squeue -u \$USER"
