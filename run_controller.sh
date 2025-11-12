#!/bin/bash

echo "Stopping any old controller or workload processes..."
# --- FIX IS HERE ---
# By removing the '-f' flag, we now match only the exact process name,
# not the full command line path. This prevents the script from killing itself.

# pkill controller || true
# pkill secondary_workload || true

echo "Starting NVIDIA Multi-Process Service (MPS) daemon..."
# This allows multiple processes to share a single GPU context efficiently
# Note: 'sudo' is used here. You may be prompted for a password.
sudo nvidia-cuda-mps-control -d

echo "Launching controller and secondary workload for GPUs 0 through 7..."

for i in {0..7}
do
  # For each GPU, launch a controller and a workload in the background (&)
  # Pin each pair to a specific GPU using CUDA_VISIBLE_DEVICES
  # Pass the --gpu-id so each pair uses a unique shared memory channel
  echo "--> Launching for GPU $i"
  CUDA_VISIBLE_DEVICES=$i ./controller --gpu-id $i &
  CUDA_VISIBLE_DEVICES=$i ./secondary_workload --gpu-id $i &
done

echo "All background processes launched. Ready to start your training script."
