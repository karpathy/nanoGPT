#!/bin/bash

echo "Shutting down the power monitoring system..."

# --- 1. Stop the Controller and Workload Processes ---
# We use the robust pkill pattern to safely find and terminate
# all instances of the controller and secondary_workload.
# The '|| true' part ensures the script doesn't exit with an
# error if the processes are already stopped.
echo "--> Terminating controller processes..."
pkill -f "[c]ontroller" || true

echo "--> Terminating secondary workload processes..."
pkill -f "[s]econdary_workload" || true


# --- 2. Stop the NVIDIA MPS Daemon ---
# The command to gracefully shut down the MPS daemon is to
# pipe the 'quit' command to its control interface.
# This requires sudo, just like starting it did.
echo "--> Shutting down NVIDIA MPS daemon..."
echo quit | sudo nvidia-cuda-mps-control


echo "Cleanup complete. All monitoring processes have been stopped."
