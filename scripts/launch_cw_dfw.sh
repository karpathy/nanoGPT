#!/bin/bash
#SBATCH -A llmservice_nemo_long-context
#SBATCH --partition batch_short
#SBATCH --nodes 2
#SBATCH -t 2:00:00
#SBATCH --exclusive # exclusive node access
#SBATCH --mem=0 # all available memory
#SBATCH --gpus-per-node=8      # n gpus per machine <required>
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8    # One task per gpu
#SBATCH --overcommit

#SBATCH --job-name=architecture

set -xu

# This script checks out a nanoGPT sha.

# The script will:
# - install a specific revision of a repo under the path /workspaces/${REPO_NAME} and add it
# to the pythonpath.
# - store the revision sha to a file under ${RESULTS_DIR}/nanogpt_git_commit
# - mount folders visible from the container in /results, /data, /workspaces and a .netrc file with credentials

# These are intended to be used as follows:
# /results will point to the <shared filesystem>/`users personal folder`/results/${WANDB_RUN_ID}/
# and will be stable across executions. This allows for restarts from checkpoints.
# /data points to the <shared filesystem>/`users personal folder` and is also stable across executions
# /code will point to a folder that changes with each job execution by appending the job id like so
# /code/${SLURM_JOB_ID}

# Job access to WANDB, github and gitlab is achieved by mounting ~/.netrc into the container.
# The user should make sure their .netrc contains the required credentials for WANDB, github and gitlab

# Usage
# sbatch --export=GIT_REF=<git sha to launch> launch_cw_dfw.sh
# Alternative usage
# ./launch_archive.sh will grab the git sha of the checked out branch and
# and substitute that for GIT_REF in the call to sbatch

# Where and what nanoGPT do we check out
# For the .netrc based authentication to succeed we must query the https endpoint rather than the ssh
NANOGPT_COMMIT_ORIGIN=https://github.com/santiagoakle/nanoGPT.git

# If WANDB_RUN_ID is not externally provided, we set it to the SLURM_JOB_ID
WANDB_RUN_ID=${WANDB_RUN_ID:-"${SLURM_JOB_ID}"}
echo "WIll report to wandb job id ${WANDB_RUN_ID}"

# Setting root directory on the cluster
USERID='sakle'
ACCOUNT_PREFIX=/lustre/fsw/portfolios/convai
TRAIN_DATA_DIR=/lustre/fsw/portfolios/convai/users/sakle/datasets/language_modelling
USER_DIR=${ACCOUNT_PREFIX}/users/${USERID}

# Container to run
# TODO (sakle) this container doesn't have any of the mamba stuff installed!
CONTAINER="nvcr.io#nvidia/nemo:24.07"

# We clone the repositories in here
RESULTS_DIR="${USER_DIR}/results/${WANDB_RUN_ID}"
# The code is stored under the git sha of the checkout
NANOGPT_CODE_DIR="${USER_DIR}/code/nanoGPT/${NANOGPT_GIT_REF}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${NANOGPT_CODE_DIR}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# Mount directories on cluster to particular paths inside the container
# Mounts a .netrc file with the WANDB credentials a results dir data dir and code dir
MOUNTS="${HOME}/.netrc:/root/.netrc,${RESULTS_DIR}:/results,${TRAIN_DATA_DIR}:/data,${NANOGPT_CODE_DIR}:/workspaces/nanoGPT"

## Rank zero of each node should do this
if [ "$SLURM_LOCALID" -eq 0 ]; then
    echo "*******STARTING********"
    echo "Checking out nanoGPT git commit ${NANOGPT_GIT_REF} for the experiment in node ${SLURM_NODEID}"
    if [ ! -d "${NANOGPT_CODE_DIR}/.git" ]; then
    # Now check out aquarium
      cd "${NANOGPT_CODE_DIR}" || exit 1
      git init
      git remote add origin ${NANOGPT_COMMIT_ORIGIN}
      git fetch origin "${NANOGPT_GIT_REF}" --depth 1
      git checkout "${NANOGPT_GIT_REF}"
    else
      echo "nanoGPT code for ${NANOGPT_GIT_REF} already cached reusing"
    fi
    echo "${NANOGPT_GIT_REF}" > "${RESULTS_DIR}/nanogpt_git_commit_job_${SLURM_JOB_ID}"
fi

# Your actual script. Note that paths are inside container.
# Change CUDA_VISIBLE_DEVICES based on your cluster node config
read -r -d '' cmd <<EOF
echo "******STARTING ${SLURM_NODEID}:${SLURM_LOCALID} ******" \
&& export PYTHONPATH=/workspaces/nanoGPT:/opt/NeMo-Framework-Launcher/launcher_scripts:/opt/megatron-lm \
&& export WANDB_TAGS="nanogpt_git_sha=$(cat ${RESULTS_DIR}/nanogpt_git_commit_job_${SLURM_JOB_ID}),SLURM_JOB_ID=${SLURM_JOB_ID}" \
&& export WANDB_RUN_ID=${WANDB_RUN_ID} \
&& cd /workspaces/nanoGPT/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py ./config/train_gpt2_cw_dfw.py
EOF

echo ${cmd}

srun -o $OUTFILE -e $ERRFILE \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  bash -c "${cmd}"
set +xu
