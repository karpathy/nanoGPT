# Makefile for common developer tasks using uv-managed environment

# Safer shell defaults
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help test unit unit-cov integration e2e acceptance test-file coverage quality quality-ext quality-ci lint format pyright mypy typecheck setup sync verify clean prepare train sample loop tensorboard deadcode gguf-help pytest-verify-layout pytest-core pytest-all check-exp check-exp-config

PYTEST_BASE=-n auto -W error --strict-markers --strict-config -v
RUN=uv run
PKG=ml_playground
VERIFY_TOOL=tools/verify_unit_test_layout.py
CLI=$(RUN) python -m $(PKG).cli
PYTEST_CMD=$(RUN) pytest $(PYTEST_BASE)

help:
	@echo "Available targets:"
	@echo "  setup        - Create venv and install all dependencies (dev + project)"
	@echo "  sync         - Sync dependencies (project + dev)"
	@echo "  test         - Run full test suite"
	@echo "  unit         - Run unit tests only"
	@echo "  unit-cov     - Run unit tests with coverage for $(PKG)"
	@echo "  integration  - Run integration tests only"
	@echo "  e2e          - Run end-to-end tests only"
	@echo "  acceptance   - Run acceptance tests only"
	@echo "  test-file    - Run a single test file: make test-file FILE=path/to/test_*.py"
	@echo "  quality      - Lint, format, type-check, and run tests"
	@echo "  quality-ext  - Extended quality: vulture + quality + mutation tests (10s cap)"
	@echo "  quality-ci   - Alias of quality-ext for CI pipelines"
	@echo "  lint         - Run Ruff checks"
	@echo "  format       - Auto-fix with Ruff and format code"
	@echo "  deadcode     - Scan for dead code with vulture"
	@echo "  prepare      - Prepare dataset: make prepare EXP=<name> [CONFIG=path]"
	@echo "  train        - Train model:   make train   EXP=<name> CONFIG=path"
	@echo "  sample       - Sample model:  make sample  EXP=<name> CONFIG=path"
	@echo "  loop         - Full loop:     make loop    EXP=<name> CONFIG=path"
	@echo "  tensorboard  - Run TensorBoard: make tensorboard LOGDIR=out/<run>/logs/tb [PORT=6006]"
	@echo "  verify       - Quick sanity check that ml_playground imports"
	@echo "  gguf-help    - Show llama.cpp converter help"
	@echo "  pyright      - Run Pyright type checker"
	@echo "  mypy         - Run Mypy type checker on ml_playground"
	@echo "  typecheck    - Run both Pyright and Mypy"
	@echo "  clean        - Remove common caches and artifacts"

setup:
	uv venv --clear && uv sync --all-groups

sync:
	uv sync --all-groups

verify:
	$(RUN) python -c "import $(PKG); print('✓ $(PKG) import OK')"

# Test targets

# Helper: verify unit test layout (if tool exists)
pytest-verify-layout:
	[ -f $(VERIFY_TOOL) ] && $(RUN) python $(VERIFY_TOOL) || true

# Helper: core pytest invoker; pass extra args with PYARGS="..."
pytest-core:
	$(PYTEST_CMD) $(PYARGS)

# Full test suite with layout verification
pytest-all: pytest-verify-layout
	$(PYTEST_CMD)

test:
	$(MAKE) pytest-all

unit: pytest-verify-layout
	$(MAKE) pytest-core PYARGS="tests/unit"

unit-cov:
	$(RUN) pytest $(PYTEST_BASE) --cov=$(PKG) --cov-report=term-missing tests/unit

integration:
	$(MAKE) pytest-core PYARGS="-m integration --no-cov"

e2e:
	$(MAKE) pytest-core PYARGS="-m e2e --no-cov"

acceptance:
	$(RUN) pytest -q -m acceptance

test-file:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-file FILE=path/to/test_*.py"; exit 2; fi; \
	$(RUN) pytest -q $(FILE)

# Quality gates

quality: format typecheck test

# Extended quality: dead code + core quality + mutation testing (non-fatal) 
quality-ext:
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
quality-ci: quality-ext

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff check --fix . && $(RUN) ruff format .

# Dead code scanning
deadcode:
	$(RUN) vulture $(PKG) --min-confidence 90

pyright:
	$(RUN) pyright

mypy:
	$(RUN) mypy --incremental $(PKG)

typecheck: pyright mypy

clean:
	# Common caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov || true
	# Python bytecode caches
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
# Parameter checks for CLI targets
check-exp:
	@if [ -z "$(EXP)" ]; then echo "Usage: set EXP=<name>"; exit 2; fi

check-exp-config:
	@if [ -z "$(EXP)" ] || [ -z "$(CONFIG)" ]; then echo "Usage: set EXP=<name> CONFIG=path"; exit 2; fi

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
prepare: check-exp
	cmd="$(CLI) prepare $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

train: check-exp-config
	$(CLI) train $(EXP) --exp-config $(CONFIG)

sample: check-exp-config
	$(CLI) sample $(EXP) --exp-config $(CONFIG)

loop: check-exp-config
	$(CLI) loop $(EXP) --exp-config $(CONFIG)

tensorboard:
	@if [ -z "$(LOGDIR)" ]; then echo "Usage: make tensorboard LOGDIR=out/<run>/logs/tb [PORT=6006]"; exit 2; fi; \
	$(RUN) tensorboard --logdir $(LOGDIR) --port $${PORT:-6006}

gguf-help:
	$(RUN) python tools/llama_cpp/convert-hf-to-gguf.py --help || true

# Coverage helper
coverage:
	$(RUN) coverage run -m pytest -m "not perf"
	$(RUN) coverage report -m
	$(RUN) coverage xml
