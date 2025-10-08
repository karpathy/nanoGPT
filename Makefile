# Thin Makefile wrappers around the Typer-based `dev-tasks` CLI.
# The canonical workflow is `uvx --from . dev-tasks ...`; these targets
# preserve muscle memory and keep CI jobs stable.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help
.SILENT:

DEV_TASK := uvx --from . dev-tasks
DEV_TASK_MUTATION := $(DEV_TASK) mutation
ARGS ?=
PYTEST_ARGS ?=
PORT ?= 6006
HOST ?= 127.0.0.1

# Support classic `make <target> <argument>` invocation for experiment-driven tasks.
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

ifeq ($(origin TOOL), undefined)
ifneq ($(filter ai-guidelines,$(firstword $(MAKECMDGOALS))),)
  TOOL := $(word 2,$(MAKECMDGOALS))
  $(eval $(TOOL):;@:)
endif
endif

CONFIG_FLAG :=
ifneq ($(strip $(CONFIG)),)
  CONFIG_FLAG := --config $(CONFIG)
endif

DRY_RUN_FLAG :=
ifeq ($(strip $(DRY_RUN)),true)
  DRY_RUN_FLAG := --dry-run
endif

LOGDIR ?=

.PHONY: all clean test
.PHONY: help setup sync verify pytest pytest-core unit unit-cov property integration e2e acceptance lint lint-check format deadcode pyright mypy typecheck quality quality-fast quality-ext coverage coverage-test coverage-report coverage-badge mutation mutation-reset mutation-summary mutation-init mutation-exec mutation-report prepare train sample loop tensorboard ai-guidelines gguf-help

all: quality

help:
	$(DEV_TASK) --help

setup:
	$(DEV_TASK) setup

sync:
	$(DEV_TASK) sync

verify:
	$(DEV_TASK) verify

pytest:
	$(DEV_TASK) pytest $(PYTEST_ARGS) $(ARGS)

pytest-core: pytest

test:
	$(DEV_TASK) test $(PYTEST_ARGS) $(ARGS)

unit:
	$(DEV_TASK) unit $(PYTEST_ARGS) $(ARGS)

unit-cov:
	$(DEV_TASK) unit-cov $(PYTEST_ARGS) $(ARGS)

property:
	$(DEV_TASK) property $(PYTEST_ARGS) $(ARGS)

integration:
	$(DEV_TASK) integration $(PYTEST_ARGS) $(ARGS)

e2e:
	$(DEV_TASK) e2e $(PYTEST_ARGS) $(ARGS)

acceptance:
	$(DEV_TASK) acceptance $(PYTEST_ARGS) $(ARGS)

lint:
	$(DEV_TASK) lint

lint-check:
	$(DEV_TASK) lint-check

format:
	$(DEV_TASK) format

deadcode:
	$(DEV_TASK) deadcode

pyright:
	$(DEV_TASK) pyright

mypy:
	$(DEV_TASK) mypy

typecheck:
	$(DEV_TASK) typecheck

quality:
	$(DEV_TASK) quality

quality-fast:
	$(DEV_TASK) quality-fast

quality-ext:
	$(DEV_TASK) quality-ext

coverage:
	$(DEV_TASK) coverage-report

coverage-test:
	$(DEV_TASK) coverage-test

coverage-report:
	$(DEV_TASK) coverage-report

coverage-badge:
	$(DEV_TASK) coverage-badge

mutation:
	$(DEV_TASK_MUTATION) run

mutation-reset:
	$(DEV_TASK_MUTATION) reset

mutation-summary:
	$(DEV_TASK_MUTATION) summary

mutation-init:
	$(DEV_TASK_MUTATION) init

mutation-exec:
	$(DEV_TASK_MUTATION) exec

mutation-report:
	$(DEV_TASK_MUTATION) report

prepare:
	@if [ -z "$(strip $(EXP))" ]; then \
		echo "Usage: make prepare EXP=<experiment> [CONFIG=path/to/config.toml]"; \
		exit 1; \
	fi
	$(DEV_TASK) prepare $(EXP) $(CONFIG_FLAG)

train:
	@if [ -z "$(strip $(EXP))" ]; then \
		echo "Usage: make train EXP=<experiment> [CONFIG=path/to/config.toml]"; \
		exit 1; \
	fi
	$(DEV_TASK) train $(EXP) $(CONFIG_FLAG)

sample:
	@if [ -z "$(strip $(EXP))" ]; then \
		echo "Usage: make sample EXP=<experiment> [CONFIG=path/to/config.toml]"; \
		exit 1; \
	fi
	$(DEV_TASK) sample $(EXP) $(CONFIG_FLAG)

loop:
	@if [ -z "$(strip $(EXP))" ]; then \
		echo "Usage: make loop EXP=<experiment> [CONFIG=path/to/config.toml]"; \
		exit 1; \
	fi
	$(DEV_TASK) loop $(EXP) $(CONFIG_FLAG)

tensorboard:
	@if [ -z "$(strip $(LOGDIR))" ]; then \
		echo "Usage: make tensorboard LOGDIR=path/to/logs [PORT=6006] [HOST=127.0.0.1]"; \
		exit 1; \
	fi
	$(DEV_TASK) tensorboard --logdir $(LOGDIR) --port $(PORT) --host $(HOST)

ai-guidelines:
	@if [ -z "$(strip $(TOOL))" ]; then \
		echo "Usage: make ai-guidelines TOOL=<name> [DRY_RUN=true]"; \
		exit 1; \
	fi
	$(DEV_TASK) ai-guidelines $(TOOL) $(DRY_RUN_FLAG)

clean:
	$(DEV_TASK) clean

gguf-help:
	$(DEV_TASK) gguf-help
