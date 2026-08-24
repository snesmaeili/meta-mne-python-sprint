#!/bin/bash
# Fetch the code the validation array job needs.
#
# A separate one-task job rather than a step on the login node: cloning
# mne-python is a few hundred MB of unpack, which belongs on a compute node.
# The array job then depends on this with afterok.
#
#   sbatch validation/v1_v2_setup.sbatch
#
#SBATCH --job-name=vardur_setup
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/sesma/logs/%x-%j.out
#SBATCH --error=/scratch/sesma/logs/%x-%j.err

set -euo pipefail

WORK=/scratch/sesma/vardur
BRANCH=epochs-variable-duration
mkdir -p "$WORK" /scratch/sesma/logs
cd "$WORK"

fetch () {  # url dir ref
  local url="$1" dir="$2" ref="$3"
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch --depth 1 origin "$ref"
    git -C "$dir" checkout -q FETCH_HEAD
  else
    git clone --depth 1 --branch "$ref" "$url" "$dir"
  fi
  echo "$dir -> $(git -C "$dir" log --oneline -1)"
}

fetch https://github.com/snesmaeili/mne-python.git mne-python "$BRANCH"
fetch https://github.com/snesmaeili/meta-mne-python-sprint.git sprint main

# the branch must be the mne that the array job imports
source /scratch/sesma/amica-python/.venv_fir/bin/activate
PYTHONPATH="$WORK/mne-python" python -c "
import mne, pathlib
print('mne', mne.__version__, 'from', pathlib.Path(mne.__file__).parent)
assert 'vardur' in mne.__file__, 'branch not on PYTHONPATH'
import numpy as np
from mne import EpochsArray, create_info
info = create_info(['a', 'b'], 100.0, 'eeg')
ep = EpochsArray([np.zeros((2, n)) for n in (71, 111)], info,
                 tmin=np.zeros(2), baseline=None, verbose=False)
assert ep.variable_duration, 'variable-duration path not active'
print('branch imports and the variable path is live')
"
echo "setup complete"
