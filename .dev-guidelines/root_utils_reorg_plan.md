# Root Utilities Reorganization Plan (P16)

## Scope
Identify and rehome root-level utilities in `ml_playground/` into coherent subpackages that mirror runtime boundaries and testing layout. No behavior changes in this PR; planning only.

## Candidates (initial inventory)
- `ml_playground/tokenizer.py` ➜ likely `ml_playground/core/tokenizer.py` or `ml_playground/models/utils/tokenizer.py`
- `ml_playground/error_handling.py` ➜ `ml_playground/core/error_handling.py`
- `ml_playground/config_loader.py` ➜ `ml_playground/configuration/loading.py`
- `ml_playground/cli.py` ➜ Already split by commands; ensure imports align with `configuration` and `training` layers
- Small helpers in root (if any) ➜ move to `core/` or closest package

## Staging approach
1. Create empty destination modules and re-export nothing (no facades).
2. Move one file per PR with import path updates and unit test fixes.
3. Keep changesets ≤200 touched lines with paired tests.
4. Run `make quality` locally; open focused PRs and merge sequentially.

## First steps (Phase 1)
- Move `error_handling.py` ➜ `ml_playground/core/error_handling.py`
  - Update imports across code/tests.
  - Validate via `make quality` and targeted tests under `tests/unit/core/`.
- Prepare `core/__init__.py` minimal (no re-exports).

## Non-goals
- No functional changes.
- No new facades in `__init__.py`.

## Validation
- `make quality` and a small targeted pytest slice for moved modules.
- CI green before merge.
