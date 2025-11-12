#!/bin/bash

# This script compiles the controller and the secondary workload.
# It's set to exit immediately if any command fails.
set -e

echo "Compiling controller (controller.cpp)..."
# Use g++ for the standard C++ controller.
# The '-lnvidia-ml' flag is crucial; it links the NVIDIA Management Library (NVML)
# which is needed for monitoring GPU utilization.
g++ controller.cpp -o controller -lnvidia-ml

echo "Compiling secondary workload (secondary_workload.cu)..."
# Use nvcc (the NVIDIA CUDA Compiler) for the .cu file.
# The '-lcublas' flag links the cuBLAS library, which provides the
# highly optimized matrix multiplication function (sgemm).
nvcc secondary_workload.cu -o secondary_workload -lcublas

echo "Compilation successful!"
echo "Executables 'controller' and 'secondary_workload' have been created."
