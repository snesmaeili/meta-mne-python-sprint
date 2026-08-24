#!/bin/bash
# V1/V2 validation of variable-duration Epochs, one array task per ds004505
# subject.
#
# Runs where the data already is: the 24 cleaned recordings are ~26 GB, so
# moving them to a workstation would be the slower half of the job.
#
# Reuses the venv the production ERSP job used and overrides only `mne`, via
# PYTHONPATH pointing at the epochs-variable-duration branch. Nothing is built.
#
#   sbatch validation/v1_v2_setup.sbatch          # fetch code (one task)
#   sbatch --dependency=afterok:<id> --array=0-23 validation/v1_v2_fir.sbatch
#
#SBATCH --job-name=vardur_validate
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/sesma/logs/%x-%A_%a.out
#SBATCH --error=/scratch/sesma/logs/%x-%A_%a.err

set -euo pipefail

VENV=/scratch/sesma/amica-python/.venv_fir
WORK=/scratch/sesma/vardur
STAGE=/scratch/sesma/results/ds004505_repl/eneuro_merged
REF=/scratch/sesma/results/ds004505_repl/eneuro_ersp
OUT=$WORK/reports

mkdir -p "$OUT" /scratch/sesma/logs

# the subject for this index, taken from what is actually staged rather than
# assumed, so a gap in the numbering cannot silently shift every task
SUBJECT=$(ls "$STAGE"/*_clean_raw.fif | sed 's#.*/##; s/_clean_raw\.fif//' \
          | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$SUBJECT" ]; then
  echo "no subject at array index ${SLURM_ARRAY_TASK_ID}; nothing to do"
  exit 0
fi
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} subject=${SUBJECT}"

source "$VENV/bin/activate"
# the branch under test wins over whatever mne the venv holds
export PYTHONPATH="$WORK/mne-python:${PYTHONPATH:-}"
export XDG_CACHE_HOME=/scratch/sesma/.cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c "import mne; print('mne', mne.__version__, 'from', mne.__file__)"

python -u "$WORK/sprint/validation/v1_v2_ds004505.py" \
  --stage-dir "$STAGE" \
  --ref-dir "$REF" \
  --subjects "$SUBJECT" \
  --out "$OUT/${SUBJECT}.json"

echo "done ${SUBJECT}"
