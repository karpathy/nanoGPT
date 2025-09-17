# Makefile for common developer tasks using uv-managed environment

# Safer shell defaults
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help
.SILENT:

.PHONY: help test unit unit-cov integration e2e acceptance test-file coverage quality quality-ext quality-ci lint format pyright mypy typecheck setup sync verify clean prepare train sample loop tensorboard deadcode gguf-help pytest-verify-layout pytest-core pytest-all check-exp check-exp-config check-tool ai-guidelines lit-setup lit lit-ephemeral-312 lit-venv-312-setup lit-venv-312 lit-docker-build lit-docker-up lit-docker-down lit-demo-penguin-uv lit-demo-glue-uv venv312-setup venv312-penguin venv312-glue venv312-lit-setup venv312-lit-run venv312-lit-stop

# Be quieter and focus output on failures only
PYTEST_BASE=-q -n auto -W error --strict-markers --strict-config
RUN=uv run
PKG=ml_playground
VERIFY_TOOL=tools/verify_unit_test_layout.py
CLI=$(RUN) python -m $(PKG).cli
PYTEST_CMD=$(RUN) pytest $(PYTEST_BASE)
TOOLS=copilot aiassistant junie kiro windsurf cursor

# Allow positional EXP argument for runtime targets, e.g.:
#   make prepare bundestag_char
#   make train bundestag_char CONFIG=path/to/config.toml
# If EXP is not explicitly set (e.g., EXP=foo), capture the second word
# of MAKECMDGOALS for these targets and convert it into EXP. Also define
# a no-op target for that second word so Make doesn't try to build it.
ifeq ($(origin EXP), undefined)
ifneq ($(filter prepare,$(firstword $(MAKECMDGOALS))),)
  EXP := $(word 2,$(MAKECMDGOALS))
  $(eval $(EXP):;@:)
endif
ifneq ($(filter train,$(firstword $(MAKECMDGOALS))),)
  EXP := $(word 2,$(MAKECMDGOALS))
  $(eval $(EXP):;@:)
endif
ifneq ($(filter sample,$(firstword $(MAKECMDGOALS))),)
  EXP := $(word 2,$(MAKECMDGOALS))
  $(eval $(EXP):;@:)
endif
ifneq ($(filter loop,$(firstword $(MAKECMDGOALS))),)
  EXP := $(word 2,$(MAKECMDGOALS))
  $(eval $(EXP):;@:)
endif
endif

help: ## Show this help
	@echo "Available targets:" && \
	echo "  vars         TOOLS=$(TOOLS)" && \
	awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST) Makefile | sort -u

setup: ## Create venv and install all dependencies (dev + project)
	uv venv --clear && uv sync --all-groups

sync: ## Sync dependencies (project + dev)
	uv sync --all-groups

verify: ## Quick sanity check that $(PKG) imports
	$(RUN) python -c "import $(PKG); print('✓ $(PKG) import OK')"

# Test targets

# Helper: verify unit test layout (if tool exists)
pytest-verify-layout: ## Verify unit test layout if verification tool exists
	[ -f $(VERIFY_TOOL) ] && $(RUN) python $(VERIFY_TOOL) || true

# Helper: core pytest invoker; pass extra args with PYARGS="..."
pytest-core: ## Run pytest with $(PYTEST_BASE); pass extra args via PYARGS
	$(PYTEST_CMD) $(PYARGS)

# Full test suite with layout verification
pytest-all: pytest-verify-layout ## Full test suite with layout verification
	$(PYTEST_CMD)

test: ## Run full test suite
	$(MAKE) pytest-all

unit: pytest-verify-layout ## Run unit tests only
	$(MAKE) pytest-core PYARGS="tests/unit"

unit-cov: ## Run unit tests with coverage for $(PKG)
	$(RUN) pytest $(PYTEST_BASE) --cov=$(PKG) --cov-report=term-missing tests/unit

integration: ## Run integration tests only
	$(MAKE) pytest-core PYARGS="-m integration --no-cov"

e2e: ## Run end-to-end tests only
	$(MAKE) pytest-core PYARGS="-m e2e --no-cov"

acceptance: ## Run acceptance tests only (quiet)
	$(RUN) pytest -q -m acceptance

test-file: ## Run a single test file: make test-file FILE=path/to/test_*.py
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-file FILE=path/to/test_*.py"; exit 2; fi; \
	$(RUN) pytest -q $(FILE)

# Quality gates

# Run type checking first to fail fast on typing issues
quality: format deadcode typecheck test ## Type-check, lint, format, and run tests

# Extended quality: dead code + core quality + mutation testing (non-fatal) 
quality-ext: ## Extended quality: vulture + quality + mutation tests (non-fatal)
	# Dead code scan (project package only)
	$(MAKE) deadcode
	# Core quality gate
	$(MAKE) quality
	# Mutation tests (non-fatal). Rely on pyproject.toml [cosmic-ray] configuration.
	set +e; \
	$(RUN) cosmic-ray init pyproject.toml out/cosmic-ray/session.sqlite >/dev/null 2>&1 || true; \
	$(RUN) cosmic-ray exec pyproject.toml out/cosmic-ray/session.sqlite; code=$$?; set -e; \
	if [ "$$code" -ne 0 ]; then \
	  echo "[warning] Cosmic Ray returned non-zero (code=$$code); proceeding"; \
	fi

# CI alias for extended quality
quality-ci: quality-ext ## Alias of quality-ext for CI pipelines

lint: ## Run Ruff checks
	$(RUN) ruff check .

format: ## Auto-fix with Ruff and format code
	$(RUN) ruff check --fix . && $(RUN) ruff format .

# Dead code scanning
deadcode: ## Scan for dead code with vulture
	$(RUN) vulture $(PKG) --min-confidence 90

pyright: ## Run Pyright type checker
	$(RUN) pyright $(PKG)

mypy: ## Run Mypy type checker on $(PKG)
	$(RUN) mypy --incremental $(PKG)

typecheck: pyright mypy ## Run both Pyright and Mypy

clean: ## Remove common caches and artifacts
	# Common caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov || true
	# Python bytecode caches
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
# Parameter checks for CLI targets
check-exp: ## Validate EXP is provided
	@if [ -z "$(EXP)" ]; then echo "Usage: set EXP=<name>"; exit 2; fi

check-exp-config: ## Validate EXP and CONFIG are provided
	@if [ -z "$(EXP)" ] || [ -z "$(CONFIG)" ]; then echo "Usage: set EXP=<name> CONFIG=path"; exit 2; fi

# Parameter check for AI guidelines tool
check-tool: ## Validate TOOL is provided (e.g., TOOL=windsurf)
	@if [ -z "$(TOOL)" ]; then echo "Usage: set TOOL=<one of: $(TOOLS)> [DRY_RUN=1]"; exit 2; fi

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
prepare: check-exp ## Prepare dataset (EXP=<name> [CONFIG=path])
	cmd="$(CLI) prepare $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	$$cmd

train: check-exp ## Train model (EXP=<name> [CONFIG=path])
	cmd="$(CLI) train $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

sample: check-exp ## Sample model (EXP=<name> [CONFIG=path])
	cmd="$(CLI) sample $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

loop: check-exp ## Full loop (EXP=<name> [CONFIG=path])
	cmd="$(CLI) loop $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

# Tools
ai-guidelines: check-tool ## Setup AI guidelines symlinks for a TOOL (TOOL=<$(TOOLS)> [DRY_RUN=1])
	cmd="$(RUN) python tools/setup_ai_guidelines.py $(TOOL)"; \
	if [ -n "$(DRY_RUN)" ]; then cmd="$$cmd --dry-run"; fi; \
	echo $$cmd; $$cmd

tensorboard: ## Run TensorBoard (LOGDIR=out/<run>/logs/tb [PORT=6006])
	@if [ -z "$(LOGDIR)" ]; then echo "Usage: make tensorboard LOGDIR=out/<run>/logs/tb [PORT=6006]"; exit 2; fi; \
	$(RUN) tensorboard --logdir $(LOGDIR) --port $${PORT:-6006}

gguf-help: ## Show llama.cpp converter help
	$(RUN) python tools/llama_cpp/convert-hf-to-gguf.py --help || true

# Coverage helper
coverage: ## Run coverage for non-performance tests and generate reports
	$(RUN) coverage run -m pytest -m "not perf"
	$(RUN) coverage report -m
	$(RUN) coverage xml

# LIT (Learning Interpretability Tool) helpers
lit-setup: ## Install optional LIT extras into the environment
	uv sync --extra lit

lit: ## Launch LIT UI for bundestag_char (override with PORT=5432, HOST=127.0.0.1)
	$(CLI) analyze bundestag_char --port $${PORT:-5432} --host $${HOST:-127.0.0.1}

# ---------------------------------------------------------------------------
# LIT: Python 3.12 isolated runs (work around NumPy 1.x vs 2.x mismatch)
# These targets avoid the project environment (Python>=3.13, NumPy>=2) and
# run LIT in a dedicated Python 3.12 context with NumPy<2.
#
# Requirements:
# - uv (https://docs.astral.sh/uv)
# - Python 3.12 will be auto-downloaded by uv if missing
#
# Usage examples:
#   make lit-ephemeral-312 PORT=5432 HOST=127.0.0.1
#   make lit-venv-312-setup  # one-time setup
#   make lit-venv-312 PORT=5432 HOST=127.0.0.1
# ---------------------------------------------------------------------------

lit-ephemeral-312: ## Launch LIT UI in an ephemeral Python 3.12 env (NumPy<2)
	# Use an isolated run that ignores the project pyproject.toml
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	uv run \
	  --no-project \
	  --python 3.12 \
	  --with "lit-nlp>=1.3.1" \
	  --with "numpy<2" \
	  -- \
	  python -m ml_playground.analysis.lit_integration --host "$${HOST}" --port "$${PORT}"

lit-venv-312-setup: ## Create persistent .venv-lit312 and install lit-nlp with NumPy<2
	uv venv --python 3.12 .venv-lit312
	uv pip install -p .venv-lit312/bin/python "lit-nlp>=1.3.1" "numpy<2"

lit-venv-312: ## Launch LIT UI using persistent .venv-lit312 (NumPy<2)
	@if [ ! -x .venv-lit312/bin/python ]; then \
	  echo "Missing .venv-lit312; run: make lit-venv-312-setup"; exit 2; \
	fi; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	.venv-lit312/bin/python -m ml_playground.analysis.lit_integration --host "$${HOST}" --port "$${PORT}"

# ---------------------------------------------------------------------------
# LIT: Official demos via uv (ephemeral, Python 3.12, NumPy<2)
# ---------------------------------------------------------------------------

lit-demo-penguin-uv: ## Run official Penguin demo (ephemeral Python 3.12)
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
	uv run \
	  --no-project \
	  --python 3.12 \
	  --with "numpy<2" \
	  --with lit-nlp==1.3.1 \
	  --with tensorflow-datasets \
	  --with tensorflow \
	  --with tf-keras \
	  -- \
	  python -m lit_nlp.examples.penguin.demo --host "$$HOST" --port "$$PORT"

lit-demo-glue-uv: ## Run official GLUE demo (ephemeral Python 3.12)
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
	uv run \
	  --no-project \
	  --python 3.12 \
	  --with "numpy<2" \
	  --with lit-nlp==1.3.1 \
	  --with tensorflow-datasets \
	  --with transformers \
	  --with sentencepiece \
	  --with tensorflow \
	  -- \
	  python -m lit_nlp.examples.glue.demo --host "$$HOST" --port "$$PORT"

# ---------------------------------------------------------------------------
# LIT: Official demos via persistent .venv312 (Python 3.12, NumPy<2)
# ---------------------------------------------------------------------------

venv312-setup: ## Create persistent .venv312 with Python 3.12 and base deps (NumPy<2)
	uv venv --python 3.12 .venv312
	uv pip install -p .venv312/bin/python "numpy<2" "lit-nlp==1.3.1"

venv312-penguin: ## Run Penguin demo using .venv312
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	printf "\n[venv312-penguin] Ensuring demo dependencies...\n"; \
	OS="$$(uname -s)"; ARCH="$$(uname -m)"; \
	if [ "$$OS" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
	  echo "[venv312-penguin] Detected macOS/arm64 -> cleaning & installing TF stack (tensorflow-macos==2.16.2 + tensorflow-metal + tf-keras)"; \
	  .venv312/bin/python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tf-keras keras >/dev/null 2>&1 || true; \
	  uv pip install -p .venv312/bin/python 'tensorflow-macos==2.16.2' tensorflow-metal tf-keras tensorflow-datasets || true; \
	else \
	  echo "[venv312-penguin] Installing tensorflow (Linux/other)"; \
	  uv pip install -p .venv312/bin/python tensorflow tensorflow-datasets tf-keras || true; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	printf "[venv312-penguin] TFDS_DATA_DIR=%s\n" "$$TFDS_DIR"; \
	printf "[venv312-penguin] Starting server on %s:%s ...\n" "$${HOST:-127.0.0.1}" "$${PORT:-5432}"; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=0 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true TFDS_DATA_DIR="$$TFDS_DIR" \
	  .venv312/bin/python -m lit_nlp.examples.penguin.demo --host "$${HOST}" --port "$${PORT}" --alsologtostderr

venv312-glue: ## Run GLUE demo using .venv312 (TensorFlow backend)
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	printf "\n[venv312-glue] Ensuring demo dependencies...\n"; \
	OS="$$(uname -s)"; ARCH="$$(uname -m)"; \
	if [ "$$OS" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
	  echo "[venv312-glue] Detected macOS/arm64 -> cleaning & installing TF stack (tensorflow-macos==2.16.2 + tensorflow-metal + tf-keras)"; \
	  .venv312/bin/python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tf-keras keras >/dev/null 2>&1 || true; \
	  uv pip install -p .venv312/bin/python 'tensorflow-macos==2.16.2' tensorflow-metal tf-keras tensorflow-datasets transformers sentencepiece || true; \
	else \
	  echo "[venv312-glue] Installing tensorflow (Linux/other)"; \
	  uv pip install -p .venv312/bin/python tensorflow tensorflow-datasets transformers sentencepiece tf-keras || true; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	printf "[venv312-glue] TFDS_DATA_DIR=%s\n" "$$TFDS_DIR"; \
	printf "[venv312-glue] Starting server on %s:%s ...\n" "$${HOST:-127.0.0.1}" "$${PORT:-5432}"; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=0 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TFDS_DATA_DIR="$$TFDS_DIR" \
	  .venv312/bin/python -m lit_nlp.examples.glue.demo --host "$${HOST}" --port "$${PORT}" --alsologtostderr

# ---------------------------------------------------------------------------
# LIT: Penguin TFDS pre-download + run (persistent .venv312)
# ---------------------------------------------------------------------------

venv312-penguin-prep: ## Pre-download TFDS 'penguins' to .cache/tensorflow_datasets
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	OS="$$(uname -s)"; ARCH="$$(uname -m)"; \
	if [ "$$OS" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
	  echo "[venv312-penguin-prep] Cleaning & installing TF stack for macOS/arm64 (tensorflow-macos==2.16.2 + tensorflow-metal + tf-keras)"; \
	  .venv312/bin/python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tf-keras keras >/dev/null 2>&1 || true; \
	  uv pip install -p .venv312/bin/python 'tensorflow-macos==2.16.2' tensorflow-metal tf-keras tensorflow-datasets || true; \
	else \
	  echo "[venv312-penguin-prep] Installing tensorflow + tensorflow-datasets"; \
	  uv pip install -p .venv312/bin/python tensorflow tensorflow-datasets tf-keras || true; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	echo "[venv312-penguin-prep] TFDS_DATA_DIR=$$TFDS_DIR"; \
	TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true \
	.venv312/bin/python -c "import os; os.environ['TFDS_DATA_DIR']='.cache/tensorflow_datasets'; print('[prep] Using TFDS_DATA_DIR=', os.environ['TFDS_DATA_DIR']); import tensorflow_datasets as tfds; b=tfds.builder('penguins', data_dir=os.environ['TFDS_DATA_DIR']); print('[prep] Downloading and preparing:', b.name); b.download_and_prepare(); print('[prep] Prepared at:', b.data_dir)"

venv312-penguin-run: ## Run Penguin demo using prepared TFDS cache
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	LIT_CACHE_DIR="$$(pwd)/.cache/lit_nlp/file_cache"; mkdir -p "$$LIT_CACHE_DIR"; \
	printf "[venv312-penguin-run] TFDS_DATA_DIR=%s\n" "$$TFDS_DIR"; \
	printf "[venv312-penguin-run] LIT_CACHE_DIR=%s\n" "$$LIT_CACHE_DIR"; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=0 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TFDS_DATA_DIR="$$TFDS_DIR" LIT_CACHE_DIR="$$LIT_CACHE_DIR" \
	  .venv312/bin/python -m lit_nlp.examples.penguin.demo --host "$${HOST}" --port "$${PORT}" --alsologtostderr

venv312-penguin-stop: ## Stop any process listening on PORT (default 5432)
	PORT="$${PORT:-5432}"; \
	PID_LIST=$$(lsof -t -iTCP:$$PORT -sTCP:LISTEN 2>/dev/null || true); \
	if [ -n "$$PID_LIST" ]; then \
	  echo "Stopping PIDs on port $$PORT: $$PID_LIST"; \
	  kill -INT $$PID_LIST || true; \
	  sleep 1; \
	  PID_LIST2=$$(lsof -t -iTCP:$$PORT -sTCP:LISTEN 2>/dev/null || true); \
	  if [ -n "$$PID_LIST2" ]; then echo "Force killing: $$PID_LIST2"; kill -TERM $$PID_LIST2 || true; fi; \
	else \
	  echo "No process listening on $$PORT"; \
	fi

# ---------------------------------------------------------------------------
# LIT: GLUE demo via persistent .venv312 (prep/run/all)
# ---------------------------------------------------------------------------

venv312-glue-prep: ## Pre-download TFDS GLUE 'sst2' to .cache/tensorflow_datasets
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	OS="$$(uname -s)"; ARCH="$$(uname -m)"; \
	if [ "$$OS" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
	  echo "[venv312-glue-prep] Cleaning & installing TF stack for macOS/arm64 (tensorflow-macos==2.16.2 + tensorflow-metal + tf-keras)"; \
	  .venv312/bin/python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tf-keras keras >/dev/null 2>&1 || true; \
	  uv pip install -p .venv312/bin/python 'tensorflow-macos==2.16.2' tensorflow-metal tf-keras tensorflow-datasets transformers sentencepiece || true; \
	else \
	  echo "[venv312-glue-prep] Installing tensorflow + extras"; \
	  uv pip install -p .venv312/bin/python tensorflow tensorflow-datasets transformers sentencepiece tf-keras || true; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	echo "[venv312-glue-prep] TFDS_DATA_DIR=$$TFDS_DIR"; \
	TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true \
	.venv312/bin/python -c "import os; os.environ['TFDS_DATA_DIR']='.cache/tensorflow_datasets'; print('[prep] Using TFDS_DATA_DIR=', os.environ['TFDS_DATA_DIR']); import tensorflow_datasets as tfds; b=tfds.builder('glue/sst2', data_dir=os.environ['TFDS_DATA_DIR']); print('[prep] Downloading and preparing:', b.name); b.download_and_prepare(); print('[prep] Prepared at:', b.data_dir)"

venv312-glue-run: ## Run GLUE demo using prepared TFDS cache (SST-2 available)
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-setup"; exit 2; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	LIT_CACHE_DIR="$$(pwd)/.cache/lit_nlp/file_cache"; mkdir -p "$$LIT_CACHE_DIR"; \
	printf "[venv312-glue-run] TFDS_DATA_DIR=%s\n" "$$TFDS_DIR"; \
	printf "[venv312-glue-run] LIT_CACHE_DIR=%s\n" "$$LIT_CACHE_DIR"; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=0 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true TFDS_DATA_DIR="$$TFDS_DIR" LIT_CACHE_DIR="$$LIT_CACHE_DIR" \
	  .venv312/bin/python -m lit_nlp.examples.glue.demo --host "$${HOST}" --port "$${PORT}" --alsologtostderr

venv312-glue-all: venv312-glue-prep venv312-glue-run ## Prepare TFDS then start GLUE demo
venv312-penguin-all: venv312-penguin-prep venv312-penguin-run ## Prepare TFDS then start Penguin demo

# ---------------------------------------------------------------------------
# LIT: Minimal integration runner (persistent .venv312)
# Uses ml_playground/analysis/lit/integration.py
# ---------------------------------------------------------------------------

venv312-lit-setup: ## Create/refresh .venv312 for LIT integration (uses constraints + platform extras)
	uv venv --python 3.12 .venv312
	uv pip install -p .venv312/bin/python -r requirements/lit-demos.constraints.txt
	OS="$$(uname -s)"; ARCH="$$(uname -m)"; \
	if [ "$$OS" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
	  .venv312/bin/python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tf-keras keras >/dev/null 2>&1 || true; \
	  uv pip install -p .venv312/bin/python 'tensorflow-macos==2.16.2' tensorflow-metal tf-keras || true; \
	else \
	  uv pip install -p .venv312/bin/python tensorflow tf-keras || true; \
	fi

venv312-lit-run: ## Run our LIT integration (bundestag_char PoC) on .venv312
	@if [ ! -x .venv312/bin/python ]; then \
	  echo "Missing .venv312; run: make venv312-lit-setup"; exit 2; \
	fi; \
	TFDS_DIR="$$(pwd)/.cache/tensorflow_datasets"; mkdir -p "$$TFDS_DIR"; \
	LIT_CACHE_DIR="$$(pwd)/.cache/lit_nlp/file_cache"; mkdir -p "$$LIT_CACHE_DIR"; \
	HOST="$${HOST:-127.0.0.1}" PORT="$${PORT:-5432}" \
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0 TF_FORCE_GPU_ALLOW_GROWTH=true TFDS_DATA_DIR="$$TFDS_DIR" LIT_CACHE_DIR="$$LIT_CACHE_DIR" \
	  .venv312/bin/python -m ml_playground.analysis.lit.integration --host "$$HOST" --port "$$PORT"

venv312-lit-stop: ## Stop LIT integration server by port (default 5432)
	PORT="$${PORT:-5432}" \
	&& .venv312/bin/python tools/port_kill.py --port "$$PORT" --graceful || true

# ---------------------------------------------------------------------------
# LIT: Dockerized service (fully encapsulated)
# Requires: Docker and Docker Compose v2 (`docker compose`)
# Files: docker/lit/Dockerfile, docker/lit/docker-compose.lit.yml
# ---------------------------------------------------------------------------

lit-docker-build: ## Build the LIT Docker image
	docker compose -f docker/lit/docker-compose.lit.yml build

lit-docker-up: ## Run the LIT Docker service (PORT=5432)
	PORT=$${PORT:-5432} docker compose -f docker/lit/docker-compose.lit.yml up --remove-orphans

lit-docker-down: ## Stop the LIT Docker service
	docker compose -f docker/lit/docker-compose.lit.yml down --remove-orphans
