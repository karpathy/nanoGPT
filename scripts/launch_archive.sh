# Uses the present git commit to launch a run in slurm

# Chooses the right script to launch
if [[ "${HOSTNAME}" == "cs-oci-ord"* ]]; then
  LAUNCH_SCRIPT=launch_oci_ord.sh
elif [[ "${HOSTNAME}" == "cw-dfw"* ]]; then
  LAUNCH_SCRIPT=launch_cw_dfw.sh
else
  exit 1
fi

# Get the top commit from the checked out branch
echo "Will submit $(git rev-parse HEAD) using ${LAUNCH_SCRIPT}"

JOB_ID_ONE=$(sbatch --parsable --export=NANOGPT_GIT_REF="$(git rev-parse HEAD)" "${LAUNCH_SCRIPT}")
echo "Queued $JOB_ID_ONE"
