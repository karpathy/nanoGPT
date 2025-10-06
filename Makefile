# Makefile for common developer tasks using uv-managed environment

# Safer shell defaults
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help
.SILENT:

.PHONY: all help test unit unit-cov property integration tests-integration e2e tests-e2e acceptance tests-acceptance test-file coverage quality quality-ext quality-ci lint lint-check format pyright mypy typecheck setup sync verify clean prepare train sample loop tensorboard deadcode gguf-help pytest-core pytest-all check-exp check-exp-config check-tool ai-guidelines venv312-lit-setup venv312-lit-run venv312-lit-stop coverage-badge
.PHONY: mutation mutation-reset mutation-summary mutation-init mutation-exec mutation-report

all: quality

# Be quieter and focus output on failures only
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export HYPOTHESIS_DATABASE_DIRECTORY := $(CURDIR)/.cache/hypothesis
export HYPOTHESIS_STORAGE_DIRECTORY := $(CURDIR)/.cache/hypothesis
PYTEST_BASE=-q -n auto -W error --strict-markers --strict-config
RUN=uv run
PKG=ml_playground
# Allow targets to select a specific interpreter for uv via RUN_PY (e.g., --python .venv312/bin/python)
RUN_PY?=
CLI=$(RUN) $(RUN_PY) python -m $(PKG).cli
PYTEST_CMD=$(RUN) pytest $(PYTEST_BASE)
TOOLS=copilot aiassistant junie kiro windsurf cursor
LIT_RUN=$(RUN) --python .venv312/bin/python

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

# Helper: core pytest invoker; pass extra args with PYARGS="..."
pytest-core: ## Run pytest with $(PYTEST_BASE); pass extra args via PYARGS
	$(PYTEST_CMD) $(PYARGS)

# Full test suite
test: ## Run full test suite
	$(PYTEST_CMD) tests

unit: ## Run unit tests only
	$(PYTEST_CMD) tests/unit

property: ## Run property-based tests only
	$(PYTEST_CMD) tests/property

unit-cov: ## Run unit tests with coverage for $(PKG)
	$(PYTEST_CMD) --cov=$(PKG) --cov-report=term-missing tests/unit

integration: ## Run integration tests only
	$(PYTEST_CMD) -m integration --no-cov

e2e: ## Run end-to-end tests only
	$(PYTEST_CMD) tests/e2e

acceptance: ## Run acceptance tests only
	$(PYTEST_CMD) tests/acceptance

# Quality gates

# Run type checking/lint/format (and tests) via pre-commit using repo hook config
quality: ## Run pre-commit (ruff, format, type checks, deadcode, tests) with .githooks config
	$(RUN) pre-commit run --config .githooks/.pre-commit-config.yaml --all-files

quality-fast: ## Run formatting + lint hooks only (fast path for local iteration)
	$(RUN) pre-commit run --config .githooks/.pre-commit-config.yaml --all-files ruff
	$(RUN) pre-commit run --config .githooks/.pre-commit-config.yaml --all-files ruff-format
	$(RUN) pre-commit run --config .githooks/.pre-commit-config.yaml --all-files mdformat

# Extended quality: core quality + mutation testing (non-fatal)
quality-ext: quality mutation ## Extended quality: vulture + quality + mutation tests (non-fatal)

mutation: ## Mutation tests (non-fatal). cosmic-ray reuses existing .cache/cosmic-ray/session.sqlite by default per pyproject.toml [cosmic-ray] config.
	$(MAKE) mutation-reset
	$(MAKE) mutation-summary
	$(MAKE) mutation-init
	$(MAKE) mutation-exec
	$(MAKE) mutation-report

mutation-reset:
	rm -f .cache/cosmic-ray/session.sqlite

mutation-summary:
	$(RUN) python tools/mutation_summary.py --config pyproject.toml

mutation-init:
	@if $(RUN) cosmic-ray init pyproject.toml .cache/cosmic-ray/session.sqlite >/dev/null 2>&1; then \
		echo "[mutation] init complete"; \
	else \
		echo "[mutation] init skipped (reusing existing session)"; \
	fi

mutation-exec:
	@echo "[mutation] starting exec"
	@$(RUN) cosmic-ray exec pyproject.toml .cache/cosmic-ray/session.sqlite || { status=$$?; echo "[warning] Cosmic Ray returned $$status"; exit $$status; }

mutation-report:
	$(RUN) python tools/mutation_report.py --config pyproject.toml

# Linting
lint: ## Lint with Ruff
	$(RUN) ruff check .
	$(RUN) ruff check --fix . && $(RUN) ruff format .

lint-check:
	$(RUN) ruff check .

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
coverage: coverage-report ## [deprecated] use `make coverage-report`
	@echo "[info] coverage-report completed; artifacts stored under .cache/coverage"
coverage-test: ## Run pytest under coverage to materialize coverage data (unit + property suites)
	mkdir -p .cache/coverage .cache/hypothesis
	dest_cov="$(CURDIR)/.cache/coverage/coverage.sqlite"
	find "$(CURDIR)/.cache/coverage" -maxdepth 1 -name 'coverage.sqlite*' -delete
	HYPOTHESIS_DATABASE_DIRECTORY=$(CURDIR)/.cache/hypothesis \
	HYPOTHESIS_STORAGE_DIRECTORY=$(CURDIR)/.cache/hypothesis \
	HYPOTHESIS_SEED=0 \
	PYTHONHASHSEED=0 \
	COVERAGE_FILE="$$dest_cov" $(RUN) coverage run -m pytest -n 0 tests/unit tests/property
	COVERAGE_FILE="$$dest_cov" $(RUN) coverage combine
	find "$(CURDIR)/.cache/coverage" -maxdepth 1 -name 'coverage.sqlite.*' -delete


coverage-report: ## Generate coverage reports (term, HTML, JSON) under .cache/coverage
	@if ! compgen -G "$(CURDIR)/.cache/coverage/coverage.sqlite*" > /dev/null; then \
		$(MAKE) coverage-test; \
	fi
	@if compgen -G "$(CURDIR)/.cache/coverage/coverage.sqlite.*" > /dev/null; then \
		COVERAGE_FILE=$(CURDIR)/.cache/coverage/coverage.sqlite $(RUN) coverage combine; \
	fi
	COVERAGE_FILE=$(CURDIR)/.cache/coverage/coverage.sqlite $(RUN) coverage report -m --fail-under=0
	COVERAGE_FILE=$(CURDIR)/.cache/coverage/coverage.sqlite $(RUN) coverage html -d .cache/coverage/htmlcov --fail-under=0
	COVERAGE_FILE=$(CURDIR)/.cache/coverage/coverage.sqlite $(RUN) coverage json -o .cache/coverage/coverage.json --fail-under=0
	COVERAGE_FILE=$(CURDIR)/.cache/coverage/coverage.sqlite $(RUN) coverage xml -o .cache/coverage/coverage.xml --fail-under=0
	@if [ "$(VERBOSE_COVERAGE)" = "1" ]; then \
		echo "[debug] coverage sqlite files:" && ls -1 .cache/coverage; \
	fi

coverage-badge: ## Generate SVG coverage badges at docs/assets
	if [ ! -f .cache/coverage/coverage.json ]; then $(MAKE) coverage-report; fi
	$(RUN) python tools/coverage_badges.py .cache/coverage/coverage.json docs/assets
venv312-lit-stop: ## Stop LIT integration server by port (default 5432)
	PORT="$${PORT:-5432}" RUN_PY="--python .venv312/bin/python" \
	&& $(RUN) $(RUN_PY) python tools/port_kill.py --port "$$PORT" --graceful || true
