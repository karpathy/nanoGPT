#!/usr/bin/env bash
# Build the obfuscated distribution.
#
# Input:  nanoGPT source tree
# Output: dist/ containing .so binaries + minimal .py support + data dirs
#
# What ships:
#   - *.so                   compiled modules (model, train, sample)
#   - configurator.py        kept as source; train/sample exec() it at runtime
#   - run_train.py           launcher shim (1 line: `import train`)
#   - run_sample.py          launcher shim
#   - bench.py               untouched (not IP-bearing)
#   - config/                training configs
#   - data/                  dataset prep scripts
#
# What does NOT ship:
#   - model.py, train.py, sample.py  (the protected IP)
#   - tests/                         (auditor-facing; kept in source tree)
#   - build artifacts
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] Clean previous build"
rm -rf dist build *.so
rm -f *.cpython-*.so

echo "[2/4] Compile .py -> .so via Cython"
.venv/bin/python setup_cython.py build_ext --inplace

echo "[3/4] Assemble dist/"
mkdir -p dist
cp -r config data dist/
cp configurator.py bench.py run_train.py run_sample.py dist/
cp *.cpython-*.so dist/
# do NOT copy model.py, train.py, sample.py

echo "[4/4] Report"
echo "--- dist/ contents ---"
find dist -maxdepth 2 -type f | sort
echo ""
echo "--- source files that would ship as .so only ---"
for f in model train sample; do
  so=$(ls dist/${f}.cpython-*.so 2>/dev/null || echo "MISSING")
  py_present="no"
  [[ -f "dist/${f}.py" ]] && py_present="YES (LEAK)"
  printf "  %-8s  %s  .py in dist: %s\n" "$f" "$so" "$py_present"
done
echo ""
echo "Build OK. Ship dist/ to customer."
