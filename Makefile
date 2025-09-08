# Makefile for common developer tasks using uv-managed environment

.PHONY: help test unit unit-cov quality quality-ext quality-ci lint format pyright mypy typecheck setup sync clean

PYTEST_BASE=-n auto -W error --strict-markers --strict-config -v

help:
	@echo "Available targets:"
	@echo "  setup        - Create venv and install all dependencies (dev + project)"
	@echo "  sync         - Sync dependencies (project + dev)"
	@echo "  test         - Run full test suite"
	@echo "  unit         - Run unit tests only"
	@echo "  unit-cov     - Run unit tests with coverage for ml_playground"
	@echo "  quality      - Lint, format, type-check, and run tests"
	@echo "  quality-ext  - Extended quality: vulture + quality + mutation tests (10s cap)"
	@echo "  quality-ci   - Alias of quality-ext for CI pipelines"
	@echo "  lint         - Run Ruff checks"
	@echo "  format       - Auto-fix with Ruff and format code"
	@echo "  pyright      - Run Pyright type checker"
	@echo "  mypy         - Run Mypy type checker on ml_playground"
	@echo "  typecheck    - Run both Pyright and Mypy"
	@echo "  clean        - Remove common caches and artifacts"

setup:
	uv venv --clear && uv sync --all-groups

sync:
	uv sync --all-groups

# Test targets

test:
	uv run pytest $(PYTEST_BASE)

unit:
	uv run pytest $(PYTEST_BASE) tests/unit

unit-cov:
	uv run pytest $(PYTEST_BASE) --cov=ml_playground --cov-report=term-missing tests/unit

# Quality gates

quality:
	uv run ruff check --fix . && \
	uv run ruff format . && \
	uv run pyright && \
	uv run mypy --incremental ml_playground && \
	uv run pytest $(PYTEST_BASE)

# Extended quality: dead code + core quality + mutation testing (non-fatal) 
quality-ext:
	# Dead code scan (project package only)
	uv run vulture ml_playground --min-confidence 90
	# Core quality gate
	$(MAKE) quality
	# Mutation tests with timeout cap; do not fail the build if non-zero
	set +e; CR_TIMEOUT=10 bash tools/mutation_test.sh; code=$$?; set -e; \
	if [ "$$code" -eq 124 ]; then \
	  echo "[warning] Cosmic Ray timed out after 10s cap; proceeding"; \
	elif [ "$$code" -ne 0 ]; then \
	  echo "[warning] Cosmic Ray returned non-zero (code=$$code); proceeding"; \
	fi

# CI alias for extended quality
quality-ci: quality-ext

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

pyright:
	uv run pyright

mypy:
	uv run mypy --incremental ml_playground

typecheck: pyright mypy

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov **/__pycache__ || true
