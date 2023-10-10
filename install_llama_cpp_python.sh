#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "This script will install llama-cpp-python with GPU support"

# Check if llama module exists, prompt to initialize submodule
if [ ! -d "$script_dir/modules/llama.cpp/models" ]; then

  read -p "llama.cpp module not initialized. Download with git? [Y/n] " response

  response=${response,,}

  if [ "$response" != "n" ]; then

    # Initialize llama submodule
    git submodule update --init --recursive

  else

    echo "Exiting. llama.cpp module required."
    exit 1

  fi

fi

read -p "Enter CUDA install location (default /usr/local/cuda): " cuda_home

cuda_home=${cuda_home:-/usr/local/cuda}

if [ ! -d "$cuda_home" ]; then
  echo "Error: $cuda_home is not a valid directory"
  exit 1
fi

export CUDA_HOME="$cuda_home"
export PATH="$cuda_home/bin:$PATH"
export LLAMA_CUBLAS=on
export LLAMA_CPP_LIB="$script_dir/modules/llama.cpp/libllama.so"

read -p "Append CUDA settings to ~/.bashrc? [Y/n] " response

response=${response,,}

if [ "$response" != "n" ]; then

  echo "export CUDA_HOME=$cuda_home" >> ~/.bashrc
  echo "export PATH=\"$cuda_home/bin:\$PATH\"" >> ~/.bashrc
  echo "export LLAMA_CUBLAS=on" >> ~/.bashrc
  echo "export LLAMA_CPP_LIB=\"$script_dir/modules/llama.cpp/libllama.so\"" >> ~/.bashrc

  echo "Appended CUDA settings to ~/.bashrc"

fi

pushd "$script_dir/modules/llama.cpp"

make clean
make libllama.so || { echo "Error compiling llama.cpp"; exit 1; }

popd

export LLAMA_CPP_LIB="$script_dir/modules/llama.cpp/libllama.so"

CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3 -m pip install llama-cpp-python --no-cache-dir

echo "llama-cpp-python installed successfully"

