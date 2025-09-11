# Makefile for common developer tasks using uv-managed environment

# Safer shell defaults
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help test unit unit-cov integration e2e acceptance test-file coverage quality quality-ext quality-ci lint format pyright mypy typecheck setup sync verify clean prepare train sample loop tensorboard deadcode gguf-help pytest-verify-layout pytest-core pytest-all check-exp check-exp-config check-tool ai-guidelines

PYTEST_BASE=-n auto -W error --strict-markers --strict-config -v
RUN=uv run
PKG=ml_playground
VERIFY_TOOL=tools/verify_unit_test_layout.py
CLI=$(RUN) python -m $(PKG).cli
PYTEST_CMD=$(RUN) pytest $(PYTEST_BASE)

help: ## Show this help
	@echo "Available targets:" && \
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

quality: format typecheck test ## Lint, format, type-check, and run tests

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
	$(RUN) pyright

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
	@if [ -z "$(TOOL)" ]; then echo "Usage: set TOOL=<one of: copilot, aiassistant, junie, kiro, windsurf, cursor> [DRY_RUN=1]"; exit 2; fi

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
prepare: check-exp ## Prepare dataset (EXP=<name> [CONFIG=path])
	cmd="$(CLI) prepare $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

train: check-exp-config ## Train model (EXP=<name> CONFIG=path)
	$(CLI) train $(EXP) --exp-config $(CONFIG)

sample: check-exp-config ## Sample model (EXP=<name> CONFIG=path)
	$(CLI) sample $(EXP) --exp-config $(CONFIG)

loop: check-exp-config ## Full loop (EXP=<name> CONFIG=path)
	$(CLI) loop $(EXP) --exp-config $(CONFIG)

# Tools
ai-guidelines: check-tool ## Setup AI guidelines symlinks for a TOOL (TOOL=<name> [DRY_RUN=1])
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
