#!/bin/bash
# Run after all array tasks in linesearch_{adam,muon}_c1.sh have finished.
# Skips completed trials via --resume and writes sweep_results.json per sweep.

set -euo pipefail

REPO_ROOT="/users/9/chen8596/nanoGPT"
CONDA_SH="/users/9/chen8596/miniconda3/etc/profile.d/conda.sh"

cd "${REPO_ROOT}"
source "${CONDA_SH}"
conda activate nanogpt

python run_sensitivity.py config/experiments/sensitivity_analysis_gpt.yaml --resume
