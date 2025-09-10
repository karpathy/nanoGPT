# Makefile for common developer tasks using uv-managed environment

.PHONY: help test unit unit-cov integration e2e acceptance test-file coverage quality quality-ext quality-ci lint format pyright mypy typecheck setup sync verify clean prepare train sample loop tensorboard deadcode gguf-help

PYTEST_BASE=-n auto -W error --strict-markers --strict-config -v

help:
	@echo "Available targets:"
	@echo "  setup        - Create venv and install all dependencies (dev + project)"
	@echo "  sync         - Sync dependencies (project + dev)"
	@echo "  test         - Run full test suite"
	@echo "  unit         - Run unit tests only"
	@echo "  unit-cov     - Run unit tests with coverage for ml_playground"
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
	uv run python -c "import ml_playground; print('✓ ml_playground import OK')"

# Test targets

test:
	[ -f tools/verify_unit_test_layout.py ] && uv run python tools/verify_unit_test_layout.py || true; \
	uv run pytest $(PYTEST_BASE)

unit:
	[ -f tools/verify_unit_test_layout.py ] && uv run python tools/verify_unit_test_layout.py || true; \
	uv run pytest $(PYTEST_BASE) tests/unit

unit-cov:
	uv run pytest $(PYTEST_BASE) --cov=ml_playground --cov-report=term-missing tests/unit

integration:
	uv run pytest $(PYTEST_BASE) -m integration --no-cov

e2e:
	uv run pytest $(PYTEST_BASE) -m e2e --no-cov

acceptance:
	uv run pytest -q -m acceptance

test-file:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-file FILE=path/to/test_*.py"; exit 2; fi; \
	uv run pytest -q $(FILE)

# Quality gates

quality:
	uv run ruff check --fix . && \
	uv run ruff format . && \
	uv run pyright && \
	uv run mypy --incremental ml_playground && \
	[ -f tools/verify_unit_test_layout.py ] && uv run python tools/verify_unit_test_layout.py || true; \
	uv run pytest $(PYTEST_BASE)

# Extended quality: dead code + core quality + mutation testing (non-fatal) 
quality-ext:
	# Dead code scan (project package only)
	$(MAKE) deadcode
	# Core quality gate
	$(MAKE) quality
	# Mutation tests (non-fatal). Rely on pyproject.toml [cosmic-ray] configuration.
	set +e; \
	uv run cosmic-ray init pyproject.toml out/cosmic-ray/session.sqlite >/dev/null 2>&1 || true; \
	uv run cosmic-ray exec pyproject.toml out/cosmic-ray/session.sqlite; code=$$?; set -e; \
	if [ "$$code" -ne 0 ]; then \
	  echo "[warning] Cosmic Ray returned non-zero (code=$$code); proceeding"; \
	fi

# CI alias for extended quality
quality-ci: quality-ext

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

# Dead code scanning
deadcode:
	uv run vulture ml_playground --min-confidence 90

pyright:
	uv run pyright

mypy:
	uv run mypy --incremental ml_playground

typecheck: pyright mypy

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov **/__pycache__ || true

# Runtime CLI wrappers (use EXP=<name> and optional CONFIG=path)
prepare:
	@if [ -z "$(EXP)" ]; then echo "Usage: make prepare EXP=<name> [CONFIG=path]"; exit 2; fi; \
	cmd="uv run python -m ml_playground.cli prepare $(EXP)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --exp-config $(CONFIG)"; fi; \
	echo $$cmd; $$cmd

train:
	@if [ -z "$(EXP)" ] || [ -z "$(CONFIG)" ]; then echo "Usage: make train EXP=<name> CONFIG=path"; exit 2; fi; \
	uv run python -m ml_playground.cli train $(EXP) --exp-config $(CONFIG)

sample:
	@if [ -z "$(EXP)" ] || [ -z "$(CONFIG)" ]; then echo "Usage: make sample EXP=<name> CONFIG=path"; exit 2; fi; \
	uv run python -m ml_playground.cli sample $(EXP) --exp-config $(CONFIG)

loop:
	@if [ -z "$(EXP)" ] || [ -z "$(CONFIG)" ]; then echo "Usage: make loop EXP=<name> CONFIG=path"; exit 2; fi; \
	uv run python -m ml_playground.cli loop $(EXP) --exp-config $(CONFIG)

tensorboard:
	@if [ -z "$(LOGDIR)" ]; then echo "Usage: make tensorboard LOGDIR=out/<run>/logs/tb [PORT=6006]"; exit 2; fi; \
	uv run tensorboard --logdir $(LOGDIR) --port $${PORT:-6006}

gguf-help:
	uv run python tools/llama_cpp/convert-hf-to-gguf.py --help || true

# Coverage helper
coverage:
	uv run coverage run -m pytest -m "not perf"
	uv run coverage report -m
	uv run coverage xml
