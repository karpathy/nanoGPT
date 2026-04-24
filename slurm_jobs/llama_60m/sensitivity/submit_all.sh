#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for rel_path in linesearch_adam_c1.sh linesearch_muon_c1.sh; do
  job_script="${SCRIPT_DIR}/${rel_path}"
  echo "Submitting ${job_script}"
  sbatch "${job_script}"
done
