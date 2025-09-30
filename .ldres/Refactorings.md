# Actionable Refactoring Todo List for nanoGPT

This document enumerates prioritized refactoring prompts that can be handed to code-generation agents. Each numbered task contains nested sub-goals to encourage small, reviewable commits.

---

## Agent-Ready, Granular Refactoring Prompts (Copy/Paste)

These prompts are self-contained and aligned with:

- `.dev-guidelines/DEVELOPMENT.md`
- `.dev-guidelines/IMPORT_GUIDELINES.md`
- `.dev-guidelines/GIT_VERSIONING.md`
- `.dev-guidelines/TESTING.md`
- `docs/framework_utilities.md`

Pre-commit hooks automatically run `make quality` during commits, so manual invocations are optional if you want faster feedback. Use short-lived feature branches, follow Conventional Commits, and always pair behavioral changes with tests. Never bypass verification (`--no-verify` is prohibited).

---

## Index

- **P1** – Stage module boundary research for `ml_playground/model.py` ✅
- **P2** – Bootstrap modular package layout and migrate model imports ✅
- **P3** – Restructure test layout to mirror new module boundaries ✅
- **P4** – Segment trainer orchestration into dedicated subpackages ✅
- **P5** – Normalize data ingestion pipeline modules ✅
- **P6** – Align configuration and CLI layering ✅
- **P7** – Retire legacy entry points after package extraction ✅
- **P8** – Reconcile test tree with canonical package layout ✅
- **P9** – Update documentation to reflect canonical package structure 🔄
- **P10** – Consolidate cache directories under `.cache/` 🔄 (small; tool configs ✅)
- **P11** – Remove mocking via dependency injection in configuration classes 🔄 (medium)
- **P12** – Add README files to key subpackages 🔄 (small)
- **P13** – Audit and fix misnamed/misplaced tests 🔄 (small)
- **P14** – Plan for `mutants/` directory management 🔄 (small; mutmut removed ✅)
- **P15** – Consolidate Python cache directories 🔄 (small, duplicate of P10; tool configs ✅)
- **P16** – Reorganize `ml_playground/` root utilities into subpackages 🔄 (large, do last)
- **P17** – Import compliance: remove re-exports and relative imports in `__init__.py` 🔄 (high)
- **P18** – Consolidate LIT integration modules and docs 🔄 (medium)

---

## Details

### P1. Stage module boundary research for `ml_playground/model.py`

**Status:** ✅ Completed (2025-09-27). See `.ldres/model_module_split_research.md` for the full findings.

- **Completed — Usage map**: Imports audited across trainers, estimators, samplers, EMA, and test suites (including mutation tests) to enumerate symbol consumers.
- **Completed — Target modules**: Candidate splits drafted for `models/core/`, `models/layers/`, and `models/optim/`, with optional utility groupings.
- **Completed — Shared utilities**: Weight-init, generation, and optimizer helpers flagged for potential relocation.
- **Completed — Dependency impacts**: Downstream pathways (training, evaluation, sampling, EMA, docs/tests) captured with required follow-up actions.

**Next reference:** Review `.ldres/model_module_split_research.md` before initiating P2 to stay aligned with the agreed layout.

### P2. Bootstrap modular package layout and migrate model imports

**Status:** ✅ Completed (2025-09-27).

- **Completed — Create package directories**: Introduced `ml_playground/models/` with `core/`, `layers/`, and `utils/` packages (empty `__init__.py` files, no re-exports).
- **Completed — Move implementations**: Split `GPT`, `GPTConfig`, layer blocks, optimizer factory, inference helpers, and weight init routines into the new submodules per `.ldres/model_module_split_research.md`.
- **Completed — Update call sites**: Migrated imports in trainers, estimators, samplers, EMA, unit/e2e tests, and mutants to the canonical modules.
- **Completed — Deprecate legacy module**: Replaced `ml_playground/model.py` with an informative shim that raises `ImportError` directing consumers to the new structure.
- **Completed — Refresh documentation**: Added the canonical import example for `ml_playground.models.core.*` in `docs/framework_utilities.md`; audit other experiment READMEs as a follow-up if they reference the legacy module path.
- **Validation**: `uv run pytest tests/unit/ml_playground/test_sampler.py tests/e2e/ml_playground/test_sample_smoke.py` now passes (14 tests, ~1s).

### P4. Segment trainer orchestration into dedicated subpackages

- **Map responsibilities**: Audit `ml_playground/trainer.py` to identify orchestration, callbacks, metrics, and checkpoint handling logic.
- **Create new packages**: Establish `ml_playground/training/loop/`, `ml_playground/training/hooks/`, and `ml_playground/training/checkpointing/` packages with minimal `__init__.py` files.
{{ ... }}
- **Enhance tests**: Mirror the package split under `tests/unit/training/` and add focused tests for callbacks/checkpoint helpers where missing.
- **Document architecture**: Extend `docs/framework_utilities.md` and any trainer-related guides to describe the new layering and extension hooks.

### P5. Normalize data ingestion pipeline modules

- **Inventory data pathways**: Review `ml_playground/data.py`, `ml_playground/prepare.py`, and `ml_playground/sampler.py` for responsibilities and shared utilities.
- **Introduce `ml_playground/data_pipeline/`**: Create subpackages for `sources/`, `transforms/`, and `sampling/`, migrating functions class-by-class with strict typing preserved.
- **Update configurations**: Adjust experiment TOML files and config builders to use the new module paths; ensure override systems (`ML_PLAYGROUND_*_OVERRIDES`) still validate correctly.
- **Strengthen testing**: Rehome relevant tests into `tests/unit/data_pipeline/` and add regression fixtures covering CLI data preparation flows.
- **Revise docs**: Update `docs/LIT.md` and any data preparation README sections to reflect new module names and usage patterns.

### P6. Align configuration and CLI layering

- **Assess coupling**: Examine `ml_playground/config.py`, `ml_playground/config_loader.py`, and `ml_playground/cli.py` for overlapping responsibilities and tight coupling.
- **Create `ml_playground/configuration/`**: Split config models, loading/parsing logic, and CLI surface adapters into separate modules with clear dependencies.
- **Refactor CLI imports**: Point CLI commands to the new configuration adapters while keeping command behavior unchanged; ensure optional dependency handling remains explicit.
- **Backfill tests**: Establish `tests/unit/configuration/` to cover parsing, validation, and CLI wiring (fast-pass tests only, per `TESTING.md`).
- **Update developer docs**: Amend `.dev-guidelines/DEVELOPMENT.md` (Configuration section) and CLI documentation to describe the new layering and extension points.

### P7. Retire legacy entry points after package extraction

#### Status checkpoint (2025-09-30)

- **Completed — Drop `ml_playground.data` shim**: Updated `ml_playground/trainer.py`, `ml_playground/sampler.py`, `ml_playground/estimator.py`, and sampler/data tests (plus mutants) to import from `ml_playground.data_pipeline.*`, then replaced `ml_playground/data.py` with a `RemovedImportError`. Validated via `uv run pytest tests/unit/ml_playground/test_sampler.py tests/unit/ml_playground/test_data_property.py`.
- **Completed — Drop `ml_playground.prepare` shim**: Added canonical pipeline in `ml_playground/data_pipeline/preparer.py`, migrated CLI, experiments, tests, mutants, and docs, and replaced `ml_playground/prepare.py` with a `RemovedImportError`. Validated via `uv run pytest tests/unit/ml_playground/test_prepare.py tests/unit/ml_playground/test_cli.py`.
- **Completed — Drop `ml_playground.trainer` shim**: Updated mutant test imports to `ml_playground.training` and `ml_playground.checkpoint`, replaced `ml_playground/trainer.py` with a `RemovedImportError`. Validated via `uv run pytest mutants/tests/e2e/ml_playground/test_train_smoke.py`.

#### Audit findings (sampler)

- **`ml_playground/sampler.py`**: **Not a shim**—houses canonical `Sampler` class and `sample()` function. CLI and experiment samplers import directly. **Action**: Keep as-is or optionally rename to `ml_playground/sampling/runner.py` for naming consistency with `preparer` (separate refactor if desired).

Remaining checklist:

- **Optional sampler rename**: If desired for consistency, plan `sampler.py` → `sampling/runner.py` move (keeping `Sampler` class), update CLI/experiments/tests, commit separately.
- **Adjust packaging artifacts**: Update `pyproject.toml`, `MANIFEST.in`, and any runtime plugin registrations so that only the new subpackages are exposed once trainer/sampler shims are retired.
- **Commit guidance**: After each shim removal or consumer update, stage only the touched module(s) plus their direct importers and commit with `git commit -m "refactor(entry-points): drop <module> shim"`.
- **Validation**: Run targeted pytest slices (trainer checkpointing, sampler smoke tests) for each subsequent shim removal to ensure no regressions.

### P8. Reconcile test tree with canonical package layout

**Status:** ✅ Completed (2025-09-30).

- **Completed — Test relocation**: Moved all test suites from `tests/unit/ml_playground/` to match canonical package layout (`tests/unit/sampling/`, `tests/unit/data_pipeline/`, `tests/unit/training/`, `tests/unit/configuration/`, `tests/unit/core/`, `tests/unit/experiments/`, `tests/unit/analysis/`).
- **Completed — Fixture consolidation**: Moved shared `conftest.py` to `tests/unit/` root.
- **Completed — Import updates**: All test imports now reference canonical modules; legacy shim dependencies removed.
- **Completed — Validation**: `uv run pytest tests/unit/` passes (209 passed, 5 skipped).

**Remaining work:**

- **Update documentation**: `.dev-guidelines/TESTING.md` still references old `tests/unit/ml_playground/` structure (lines 52, 249).
- **Update other docs**: Check all markdown files for outdated references to legacy shims and test paths.

### P9. Update documentation to reflect canonical package structure

**Status:** 🔄 Planned (2025-09-30).

**Objective**: Ensure all documentation reflects the completed refactoring (retired shims, canonical package layout, reorganized test tree).

**Files requiring updates:**

1. **`.dev-guidelines/TESTING.md`**
   - Line 52: Update `tests/unit/ml_playground/test_<module>.py` → `tests/unit/<package>/test_<module>.py`
   - Line 249: Update example path to reflect new structure
   - Add quick reference table showing canonical test layout

2. **`.dev-guidelines/REQUIREMENTS.md`**
   - Check for any references to `ml_playground.model`, `ml_playground.data`, etc.

3. **`.dev-guidelines/SETUP.md`**
   - Check for references to `ml_playground.prepare`

4. **`docs/framework_utilities.md`**
   - Already updated (2025-09-30) but verify completeness

5. **`ml_playground/experiments/Readme.md`**
   - Already updated (2025-09-30) but verify completeness

6. **Experiment-specific READMEs**
   - `ml_playground/experiments/shakespeare/Readme.md`
   - `ml_playground/experiments/bundestag_char/Readme.md`
   - `ml_playground/experiments/bundestag_tiktoken/Readme.md`
   - `ml_playground/experiments/bundestag_qwen15b_lora_mps/Readme.md`
   - `ml_playground/experiments/speakger/Readme.md`
   - Check for outdated import examples or references to legacy modules

7. **Test READMEs**
   - `tests/unit/README.md`
   - `tests/integration/README.md`
   - `tests/e2e/README.md`
   - Update to reflect new test tree structure

8. **Root documentation**
   - `README.md` - Check for outdated module references
   - `AGENT.md` - Check for outdated examples

**Action items:**

1. **Audit all markdown files** for references to:
   - `ml_playground.data` → should be `ml_playground.data_pipeline`
   - `ml_playground.prepare` → should be `ml_playground.data_pipeline.preparer` or `ml_playground.data_pipeline`
   - `ml_playground.trainer` → should be `ml_playground.training`
   - `ml_playground.sampler` → should be `ml_playground.sampling`
   - `ml_playground.model` → should be `ml_playground.models.core.model`
   - `ml_playground.config` → should be `ml_playground.configuration`
   - `ml_playground.config_loader` → should be `ml_playground.configuration.loading`
   - `tests/unit/ml_playground/` → should be `tests/unit/<package>/`

2. **Update each file** with canonical references, preserving examples and clarity.

3. **Add migration notes** where helpful (e.g., "Previously `ml_playground.prepare`, now `ml_playground.data_pipeline`").

4. **Commit strategy**: One commit per documentation area (e.g., `docs(testing): update TESTING.md for canonical test layout`, `docs(experiments): update experiment READMEs for canonical imports`).

5. **Validation**: Run `make verify` to ensure no broken links or import examples in docs.

**Commit guidance**: `git commit -m "docs(<area>): update for canonical package structure"`

---

### P10. Consolidate cache directories under `.cache/`

**Status:** 🔄 Partially completed (2025-09-30).

**Objective**: Move all cache directories to a single `.cache/` root to improve project organization and simplify gitignore rules.

**Priority**: Small - can be done independently

**Current cache locations:**

- `.pytest_cache/` → `.cache/pytest/`
- `.hypothesis/` → `.cache/hypothesis/`
- `.mypy_cache/` → `.cache/mypy/`
- `.ruff_cache/` → `.cache/ruff/`
- `.uv_cache/` → `.cache/uv/`
- `ml_playground/.hf_cache/` → `.cache/huggingface/`
- `__pycache__/` directories → keep as-is (Python standard)

**Action items:**

1. **Update tool configurations** to point to new cache locations:
   - `pyproject.toml`: Update `[tool.pytest]`, `[tool.mypy]`, `[tool.ruff]` cache paths
   - Check if UV supports custom cache location via env var or config

2. **Update `.gitignore`**:
   - Simplify to single `.cache/` pattern
   - Keep `__pycache__/` as separate pattern (Python standard)
   - Remove redundant individual cache patterns

3. **Create migration script** (optional):
   - Script to move existing caches to new locations
   - Or document manual cleanup: `rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache .uv_cache ml_playground/.hf_cache`

4. **Update documentation**:
   - `.dev-guidelines/SETUP.md` - Document new cache structure
   - `README.md` - Update any cache-related instructions

**Commit guidance**: `git commit -m "chore(cache): consolidate all caches under .cache/"`

---

### P11. Remove mocking via dependency injection in configuration classes

**Status:** 🔄 Planned (2025-09-30).

**Objective**: Eliminate test mocking by injecting all dependencies (functions, clients, file operations) through configuration classes. This aligns with the validated configuration-based construction approach and makes tests more explicit and maintainable.

**Priority**: Medium - improves testability and code clarity

**Current problem:**

- Tests heavily use `mocker.patch()`, `monkeypatch`, and `@pytest.mock` to replace functions
- Makes tests brittle and coupled to implementation details
- Hard to reason about what dependencies a component actually needs
- Mocking hides the true dependency graph

**Target solution:**

- All external dependencies (functions, I/O, network, time) injected via configuration classes
- Configuration classes become the single source of truth for dependencies
- Tests provide alternative implementations via configuration, not mocking
- Production code uses default/real implementations from configuration

**Major areas with mocking (from grep analysis):**

1. **CLI tests** (`test_cli.py` - 42 mocks)
   - Mocks: `_run_prepare`, `_run_train`, `_run_sample`, config loaders
   - Solution: Add function fields to config classes

2. **Checkpoint tests** (`test_checkpoint.py` - 31 mocks)
   - Mocks: File I/O, torch.save, torch.load, path operations
   - Solution: Add I/O abstraction to `RuntimeConfig` or create `CheckpointingConfig`

3. **Integration tests** (`test_datasets_shakespeare.py` - 29 mocks)
   - Mocks: HTTP requests, file downloads, path operations
   - Solution: Add download/fetch functions to `PreparerConfig`

4. **Training loop tests** (`test_training_runner.py` - 28 mocks)
   - Mocks: Model step, optimizer step, checkpoint save, eval functions
   - Solution: Add hooks/callbacks to `TrainerConfig`

5. **Experiments loader tests** (`test_experiments_loader.py` - 20 mocks)
   - Mocks: Module imports, preparer instantiation
   - Solution: Add factory functions to experiment configs

6. **Sampler tests** (`test_runner.py` - 15 mocks)
   - Mocks: Model forward, checkpoint loading, tokenizer
   - Solution: Add inference functions to `SamplerConfig`

**Implementation strategy:**

**Phase 1: Add dependency fields to configuration classes**

1. **`PreparerConfig`** - Add fields for:

   ```python
   class PreparerConfig(_FrozenStrictModel):
       # ... existing fields ...
       
       # Dependency injection fields (with sensible defaults)
       fetch_fn: Callable[[str, Path], None] = Field(default=download_file)
       read_fn: Callable[[Path], str] = Field(default=read_text_file)
       write_fn: Callable[[Path, bytes], None] = Field(default=write_binary_file)
       tokenizer_factory: Callable[[str], Tokenizer] = Field(default=create_tokenizer)
   ```

2. **`TrainerConfig`** - Add fields for:

   ```python
   class TrainerConfig(_FrozenStrictModel):
       # ... existing fields ...
       
       # Training hooks
       before_step_hook: Callable[[int], None] | None = None
       after_step_hook: Callable[[int, dict], None] | None = None
       checkpoint_save_fn: Callable[[Path, dict], None] = Field(default=torch.save)
       checkpoint_load_fn: Callable[[Path], dict] = Field(default=torch.load)
   ```

3. **`SamplerConfig`** - Add fields for:

   ```python
   class SamplerConfig(_FrozenStrictModel):
       # ... existing fields ...
       
       # Inference dependencies
       checkpoint_load_fn: Callable[[Path], dict] = Field(default=load_checkpoint)
       model_factory: Callable[[ModelConfig], nn.Module] = Field(default=create_model)
   ```

4. **`RuntimeConfig`** - Add fields for:

   ```python
   class RuntimeConfig(_FrozenStrictModel):
       # ... existing fields ...
       
       # I/O and system dependencies
       path_exists_fn: Callable[[Path], bool] = Field(default=lambda p: p.exists())
       makedirs_fn: Callable[[Path], None] = Field(default=lambda p: p.mkdir(parents=True, exist_ok=True))
       get_time_fn: Callable[[], float] = Field(default=time.time)
   ```

**Phase 2: Update production code to use injected dependencies**

1. Update preparer implementations to use `config.fetch_fn`, `config.read_fn`, etc.
2. Update training loop to use `config.checkpoint_save_fn`, hooks, etc.
3. Update sampler to use `config.checkpoint_load_fn`, `config.model_factory`, etc.

**Phase 3: Update tests to inject test implementations**

Before (with mocking):

```python
def test_prepare_downloads_file(mocker):
    mock_download = mocker.patch('module.download_file')
    preparer.prepare()
    mock_download.assert_called_once()
```

After (with dependency injection):

```python
def test_prepare_downloads_file():
    downloads = []
    def fake_download(url: str, path: Path) -> None:
        downloads.append((url, path))
    
    config = PreparerConfig(..., fetch_fn=fake_download)
    preparer = Preparer(config)
    preparer.prepare()
    
    assert len(downloads) == 1
```

**Phase 4: Remove mocking infrastructure**

1. Remove `pytest-mock` from dependencies (if no longer needed elsewhere)
2. Remove all `mocker` fixtures from tests
3. Remove all `monkeypatch` usage for function replacement
4. Update `.dev-guidelines/TESTING.md` to document dependency injection pattern

**Pydantic considerations:**

- Callable fields need special handling in Pydantic
- Use `Field(exclude=True)` for callables to avoid serialization issues
- Or use `ConfigDict(arbitrary_types_allowed=True)`
- Document that configs with custom callables cannot be serialized to TOML

**Action items:**

1. **Audit current mocking usage**: Document all mocked functions and their purposes
2. **Design dependency injection API**: Plan config class extensions
3. **Implement in phases**: Start with one config class (e.g., `PreparerConfig`)
4. **Update production code**: Use injected dependencies
5. **Update tests**: Replace mocks with injected test implementations
6. **Document pattern**: Add examples to testing guidelines

**Commit strategy**:

- `refactor(config): add dependency injection to PreparerConfig`
- `refactor(preparer): use injected dependencies from config`
- `test(preparer): replace mocks with dependency injection`
- Repeat for each config class

**Validation**: All tests should pass without mocking after each phase.

---

### P12. Add README files to key subpackages

**Status:** 🔄 Planned (2025-09-30).

**Objective**: Improve discoverability and onboarding by adding README files to major subpackages explaining their purpose, structure, and key APIs.

**Priority**: Small - documentation improvement

**Packages needing READMEs:**

1. **`ml_playground/training/README.md`**
   - Purpose: Training loop orchestration, checkpointing, hooks, LR scheduling
   - Key modules: `loop/`, `checkpointing/`, `hooks/`, `lr_scheduler.py`, `ema.py`
   - Entry points: `Trainer`, `train()`, checkpoint management

2. **`ml_playground/data_pipeline/README.md`**
   - Purpose: Data preparation, tokenization, batch sampling
   - Key modules: `transforms/`, `sources/`, `sampling/`, `preparer.py`
   - Entry points: `create_pipeline()`, `prepare_with_tokenizer()`, `SimpleBatches`

3. **`ml_playground/sampling/README.md`**
   - Purpose: Model inference and text generation
   - Key modules: `runner.py`
   - Entry points: `Sampler`, `sample()`

4. **`ml_playground/configuration/README.md`**
   - Purpose: Configuration models and loading
   - Key modules: `models.py`, `loading.py`, `cli.py`
   - Entry points: Configuration classes, `load_full_experiment_config()`

5. **`ml_playground/models/README.md`**
   - Purpose: Model architecture and layers
   - Key modules: `core/`, `layers/`, `utils/`
   - Entry points: `GPT`, layer components

6. **`ml_playground/core/README.md`** (if created in P11)
   - Purpose: Core utilities (error handling, tokenization, protocols)
   - Key modules: `error_handling.py`, `tokenizer.py`, `logging_protocol.py`

**README template:**

```markdown
# <Package Name>

## Purpose
[Brief description of what this package does]

## Structure
- `module1.py` - [Description]
- `module2.py` - [Description]
- `subpackage/` - [Description]

## Key APIs
- `ClassName` - [Description]
- `function_name()` - [Description]

## Usage Example
\`\`\`python
from ml_playground.<package> import ...
\`\`\`

## Related Documentation
- [Link to relevant docs]
```

**Commit guidance**: `git commit -m "docs(<package>): add README explaining structure and APIs"`

---

### P13. Audit and fix misnamed/misplaced tests

**Status:** 🔄 Planned (2025-09-30).

**Objective**: Ensure all tests follow naming conventions and are placed in the correct directories matching the code they test.

**Priority**: Small - improves test organization

**Naming conventions (from `.dev-guidelines/TESTING.md`):**

- Test files: `test_<module>.py`
- Test functions: `test_<behavior>_<condition>_<expected>()`
- Property-based tests: `test_<module>_property.py`

**Action items:**

1. **Audit test names**:
   - Find tests not following `test_*.py` pattern
   - Find test functions not following `test_*` pattern
   - Check for overly generic names (e.g., `test_basic`, `test_smoke`)

2. **Audit test placement**:
   - Verify each test file is in the correct package directory
   - Example: `test_checkpoint.py` should be in `tests/unit/training/checkpointing/` (not `tests/unit/core/`)
   - Check for tests in wrong level (unit vs integration vs e2e)

3. **Check for duplicate test names** across different files (pytest allows but can be confusing)

4. **Verify test docstrings**: Each test should have a one-line docstring explaining behavior

**Known issues to investigate:**

- `tests/unit/core/test_checkpoint.py` - Should this be in `tests/unit/training/checkpointing/`?
- `tests/unit/core/test_tokenizer.py` - Correct location after P11 tokenizer move?
- Any tests still referencing legacy `ml_playground` structure

**Commit guidance**: `git commit -m "test(structure): rename/relocate misplaced tests"`

---

### P14. Plan for `mutants/` directory management

**Status:** 🔄 Partially completed (2025-09-30).

**Objective**: Clarify the purpose and management strategy for the `mutants/` directory (mutation testing artifacts).

**Priority**: Small - organizational clarity

**Current state:**

- `mutants/` is gitignored (line 41 in `.gitignore`)
- Contains test copies and mutation testing artifacts
- Appears to be generated by cosmic-ray or similar mutation testing tool
- Takes up space and can cause confusion

**Completed:**

- Removed `mutmut` configuration from `pyproject.toml`.
- Repointed `cosmic-ray` to a live module and tests (`ml_playground/checkpoint.py`, `tests/unit/training/checkpointing/test_service.py`).

**Options:**

**Option A: Keep as gitignored build artifact (RECOMMENDED)**

- Treat `mutants/` as a build artifact like `__pycache__` or `.pytest_cache`
- Ensure it's properly gitignored (already done)
- Document in `.dev-guidelines/TESTING.md`:
  - What `mutants/` is (mutation testing artifacts)
  - How it's generated (`cosmic-ray` tool)
  - How to clean it up (`make clean` or `rm -rf mutants/`)
  - That it should never be committed

**Option B: Move to `.cache/mutants/`**

- Consolidate with other build artifacts
- Update cosmic-ray configuration to output to `.cache/mutants/`
- Update `.gitignore` accordingly

**Option C: Remove mutation testing**

- If mutation testing is not actively used, remove cosmic-ray dependency
- Remove `mutants/` directory
- Remove cosmic-ray from `pyproject.toml` dev dependencies

**Action items:**

1. **Verify cosmic-ray usage**:
   - Check if mutation testing is run in CI
   - Check if it's documented in testing guidelines
   - Determine if it's actively maintained

2. **Choose option** based on usage

3. **Update documentation**:
   - `.dev-guidelines/TESTING.md` - Document mutation testing workflow
   - Add to "Project artifacts and caches" section
   - Explain cleanup process

4. **Update Makefile** (if needed):
   - Add `make mutants` target for running mutation tests
   - Add cleanup to `make clean`

5. **Consider renaming** to `.mutants/` (hidden directory) if keeping as build artifact

**Commit guidance**: `git commit -m "docs(testing): document mutation testing artifacts in mutants/"`

---

### P17. Import compliance: remove re-exports and relative imports in `__init__.py`

**Status:** ✅ Completed (2025-09-30).

**Objective**: Achieve 100% compliance with `IMPORT_GUIDELINES.md` by removing re-export facades and relative imports in package `__init__.py` files.

**Scope:**
- Replace relative imports with absolute ones in `ml_playground/configuration/cli.py`.
- Remove (or minimize) re-exports in:
  - `ml_playground/training/__init__.py`
  - `ml_playground/training/loop/__init__.py`
  - `ml_playground/training/checkpointing/__init__.py`
  - `ml_playground/training/hooks/__init__.py`
  - `ml_playground/sampling/__init__.py`
  - `ml_playground/configuration/__init__.py` (phase removal; update imports first)
  - `ml_playground/data_pipeline/__init__.py`
  - `ml_playground/data_pipeline/sampling/__init__.py`
  - `ml_playground/data_pipeline/transforms/__init__.py`

**Completed:**
- ✅ Fixed relative imports in configuration/cli.py
- ✅ Updated consumers to import from submodules (training.loop.runner, sampling.runner, etc.)
- ✅ Removed all re-exports from __init__.py files
- ✅ Updated CLI, experiments, and tests to use direct imports
- ✅ Achieved zero backwards compatibility
- ✅ All quality gates pass: linting, type checking, full test suite

**Commit**: `ab45200 refactor(import-compliance): complete P17 - remove all re-exports and relative imports`

**PR**: #8 - Merged into master

---

### P18. Consolidate LIT integration modules and docs

**Status:** ✅ Completed (2025-09-30).

**Objective**: Keep a single canonical LIT integration module and align docs.

**Scope:**
- Prefer `ml_playground.analysis.lit.integration` as canonical.
- Remove `ml_playground/analysis/lit_integration.py` (legacy duplicate) and update references.
- Update `docs/LIT.md` Make targets and invocations to the canonical module.

**Completed:**
- ✅ Removed legacy `ml_playground/analysis/lit_integration.py` file (241 lines)
- ✅ Kept canonical `ml_playground/analysis/lit/integration.py` (273 lines)  
- ✅ Updated public API policy test to remove forbidden import check for legacy file
- ✅ Verified Makefile targets already use canonical module
- ✅ No docs updates needed - docs already reference correct paths

**Commit**: `refactor(analysis): consolidate LIT integration; remove legacy duplicate`

### P15. Consolidate Python cache directories (`.mypy_cache/`, `.ruff_cache/`, etc.)

**Status:** 🔄 Partially completed (2025-09-30). Duplicate of P10 — tracked there.

**Objective**: Configure Python tools to use `.cache/` subdirectories instead of root-level cache directories.

**Priority**: Small - organizational improvement (duplicate of P10, can be merged)

**Tools to configure:**

1. **pytest** → `.cache/pytest/`

   ```toml
   [tool.pytest.ini_options]
   cache_dir = ".cache/pytest"
   ```

2. **mypy** → `.cache/mypy/`

   ```toml
   [tool.mypy]
   cache_dir = ".cache/mypy"
   ```

3. **ruff** → `.cache/ruff/`

   ```toml
   [tool.ruff]
   cache-dir = ".cache/ruff"
   ```

4. **hypothesis** → `.cache/hypothesis/`

   ```toml
   [tool.pytest.ini_options]
   hypothesis_storage_directory = ".cache/hypothesis"
   ```

5. **uv** → `.cache/uv/`
   - Check UV documentation for cache configuration
   - May require environment variable: `UV_CACHE_DIR=.cache/uv`

**Completed:**

1. **`pyproject.toml`** updated with cache directory configurations (see P10)

**Action items:**
2. **Update `.gitignore`**: Replace individual cache patterns with single `.cache/` entry
3. **Clean up old caches**: `rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache .uv_cache`
4. **Test tools** to ensure they create caches in new locations
5. **Update CI configuration** if it references old cache paths

**Commit guidance**: `git commit -m "chore(config): consolidate tool caches under .cache/"`

---

### P16. Reorganize `ml_playground/` root utilities into subpackages

**Status:** 🔄 Planned (2025-09-30).

**Objective**: Reduce clutter in `ml_playground/` root by moving utilities into logical subpackages. Keep only `cli.py` and `__init__.py` at root level (CLI is the main entry point; future entry points like REST/MCP will be added later).

**Priority**: Large - significant refactoring (should be done last after other improvements)

**Current root files to relocate:**

1. **Checkpoint management** → `ml_playground/training/checkpointing/`
   - `checkpoint.py` → `ml_playground/training/checkpointing/manager.py`
   - Update imports in training loop, sampler, tests
   - Note: Already has checkpointing service, consolidate with checkpoint.py

2. **Error handling** → `ml_playground/core/`
   - `error_handling.py` → `ml_playground/core/error_handling.py`
   - Update imports across codebase (widely used)

3. **Tokenization** → `ml_playground/core/`
   - `tokenizer.py` → `ml_playground/core/tokenizer.py`
   - `tokenizer_protocol.py` → `ml_playground/core/tokenizer_protocol.py`
   - Update imports in data pipeline, experiments, tests
   - Note: Memory shows relative import violation in tokenizer.py to fix

4. **Training utilities** → `ml_playground/training/`
   - `lr_scheduler.py` → `ml_playground/training/lr_scheduler.py`
   - `ema.py` → `ml_playground/training/ema.py`
   - `estimator.py` → `ml_playground/training/estimator.py`
   - Update imports in training loop, tests

5. **Protocols** → `ml_playground/core/`
   - `logging_protocol.py` → `ml_playground/core/logging_protocol.py`
   - Update imports across codebase

6. **Internal utilities** → `ml_playground/core/` or remove
   - `_file_state.py` → `ml_playground/core/_file_state.py` (or inline if only used in one place)

**Files to keep at root:**

- `cli.py` - Main CLI entry point
- `__init__.py` - Package initialization

**Action items:**

1. **Create `ml_playground/core/` package** if it doesn't exist
   - Add `__init__.py`
   - Plan exports for common utilities

2. **Move files using `git mv`** to preserve history
   - One logical group per commit
   - Update test files in same commit as source

3. **Update all imports** across:
   - Runtime code (`training/`, `sampling/`, `data_pipeline/`, `experiments/`)
   - Tests (`tests/unit/`, `tests/integration/`, `tests/e2e/`)
   - Mutants (`mutants/tests/`)
   - Update test locations to match (e.g., `tests/unit/core/test_checkpoint.py` → `tests/unit/training/checkpointing/test_manager.py`)

4. **Fix import violations while moving**:
   - Add `from __future__ import annotations` if missing
   - Change relative imports to absolute imports (per memory IMPORT_GUIDELINES violations)

5. **Update `__init__.py`** exports:
   - `ml_playground/__init__.py` - Re-export commonly used items for backward compatibility
   - Consider adding deprecation warnings for old import paths

6. **Update documentation**:
   - Update P9 documentation task to reflect new locations
   - Update `docs/framework_utilities.md` with new import paths

**Commit strategy**:

1. `refactor(core): create core package and move error_handling + protocols`
2. `refactor(core): move tokenizer modules to core package`
3. `refactor(training): move lr_scheduler + ema + estimator to training`
4. `refactor(training): consolidate checkpoint.py into checkpointing/manager.py`
5. `refactor(cleanup): update __init__.py exports for new structure`

**Validation**: `uv run pytest tests/` should pass after each move.

**Dependencies**: Should be done AFTER:

- P9 (Documentation updates)
- P11 (Dependency injection) - reduces mocking that might break during moves
- P13 (Test organization) - ensures tests are in right place

**Benefits**:

- Cleaner package root
- Logical grouping of related utilities
- Easier to find and understand module purposes
- Prepares for future entry points (REST API, MCP server)
- Aligns with canonical package structure established in P2-P8
