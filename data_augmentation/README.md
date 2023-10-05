# Running Mistral 7B Test

Scripts here download and test the Mistral 7B model with a python wrapper for
`llama.cpp` called `llama-cpp-python`.

## Install Steps

1. First install the nanogpt requirements (see main [README.md](../README.md))
2. Second install `llama-cpp-python` and dependencies via the installation
   script provided in the repo root directory:

```bash
bash install_llama_cpp_python.sh
```
3. Finally cd into this directory and run the `download_and_test_mistral.sh`
   script via sourcing (b/c will need one's python environment):

```bash
source download_and_test_mistral7b.sh
```

This script will download mistral7b if not already in the `./models` directory,
and start the `llama-cpp-python_example.py` script.

This will should complete fairly quickly with GPU acceleration.
