# Branch Coverage Uplift Plan

## Baseline and policy guardrails

- Current branch coverage reported by the badge is **78.33%**, well below the 100% expectation we set for the ml_playground package.【F:docs/assets/coverage-branches.svg†L1-L14】【F:.dev-guidelines/TESTING.md†L203-L218】 Line coverage sits at **87.28%**, so the remaining work is overwhelmingly branch-specific rather than missing features.【F:docs/coverage/roadmap.md†L5-L39】
- The existing coverage roadmap already calls out deterministic modules (e.g., `cli.py`, tokenization transforms, checkpointing services), protocol surfaces, and GPU-dependent helpers as the remaining gaps before we can ratchet the thresholds upward.【F:docs/coverage/roadmap.md†L40-L61】
- The coverage opportunities log details exactly which lines and behaviors are still uncovered, giving us an actionable backlog to close branch gaps without introducing mocks or test-only code paths.【F:.ldres/coverage-opportunities.md†L31-L160】

## Prioritized targets and expected branch gains

The estimates below translate the uncovered branches into projected gains toward the 21.67 percentage points we need to reach 100% branch coverage. After reconciling the counts with the opportunities log, the realistic uplift is **≈17–18 percentage points**, leaving a narrow margin for hardware-specific paths that will likely need explicit skips. Every target references concrete control-flow decisions so we can design deterministic tests that comply with the no-mocking rule in the testing guidelines.【F:.dev-guidelines/TESTING.md†L155-L177】

### 1. Harden the Typer CLI surface (≈ +6.5–7.0 pp)

- **Global options error handling (≈ +1.5 pp)** – `--exp-config` pointing at a missing file exits with code 2; context initialization falls back for bad Typer context objects.【F:src/ml_playground/cli.py†L361-L391】 Exercise `global_options` with Typer’s runner using tmp configs and malformed contexts.
- **Command runners (≈ +2.0 pp)** – `prepare`, `train`, and `sample` rely on `run_or_exit` to surface interrupts and domain exceptions.【F:src/ml_playground/cli.py†L188-L441】 Inject callables that raise `KeyboardInterrupt`, `FileNotFoundError`, and `ValueError`, and assert Typer exit codes plus logging.
- **Device setup fallbacks (≈ +1.2 pp)** – `_global_device_setup` swallows torch errors and checks CUDA availability via injected helpers.【F:src/ml_playground/cli.py†L137-L162】 Provide fake `cuda_is_available` and torch-like callables that raise `RuntimeError` to cover both paths.
- **Analysis guard rails (≈ +1.0 pp)** – `_run_analyze` rejects unsupported experiments before logging placeholders.【F:src/ml_playground/cli.py†L320-L333】 Assert the runtime error for non-`bundestag_char` experiments and the info log path when the experiment matches.
- **Directory logging resilience (≈ +0.8 pp)** – `_log_dir` and `_log_command_status` distinguish between unset, missing, and existing paths while swallowing filesystem errors.【F:src/ml_playground/cli.py†L222-L251】 Use tmp directories, including permission-denied cases, to walk every branch without monkeypatching.

### 2. Cover tokenizer metadata edge cases (≈ +2.0 pp)

- `prepare_with_tokenizer` rebuilds vocabularies differently for char vs. word tokenizers and includes empty-input fallbacks.【F:src/ml_playground/data_pipeline/transforms/tokenization.py†L39-L63】 Add parametrized/property tests (e.g., `hypothesis`) that feed empty strings and mixed punctuation to hit both sides of the `if words` check.
- `create_standardized_metadata` guards lookups for `stoi` and `encoding_name`, suppressing attribute errors when metadata is missing.【F:src/ml_playground/data_pipeline/transforms/tokenization.py†L74-L104】 Inject minimal fake tokenizer objects to exercise both the guarded lookups and the exception handler.

### 3. Retrofit runtime seeding with injectable seams (≈ +2.5–3.0 pp)

- `setup_runtime` currently calls torch CUDA helpers directly, leaving the CUDA-available path, CUDA error handler, and autocast branch uncovered in CPU-only CI.【F:src/ml_playground/training/hooks/runtime.py†L33-L51】 Introduce optional callables for CUDA availability and autocast creation so tests can trigger both the CUDA and CPU flows without violating the “no mocks” guideline. Two focused tests (GPU happy path, CUDA failure raising `RuntimeError`) should recover the missing branches.

### 4. Close checkpointing defensive paths (≈ +3.5–4.0 pp)

- `checkpoint_manager.py` still misses branches around `missing_ok`, best-checkpoint selection, and sidecar cleanup.【F:.ldres/coverage-opportunities.md†L37-L48】 Build tmp-directory tests that create and remove fake checkpoint files to exercise the `FileNotFoundError`, validation failure, and cleanup branches.
- `checkpointing/service.py` has dependency-injection override error handling paths left uncovered.【F:.ldres/coverage-opportunities.md†L31-L36】 Supply a failing override to trigger the defensive branch and confirm logging without mocks.

### 5. Finish core inference and experiment prep edge cases (≈ +2.0–2.5 pp)

- `models/core/inference.py` lacks branch coverage for AttributeError paths when configs are missing.【F:.ldres/coverage-opportunities.md†L59-L66】 Instantiate the inference helper with a minimal stub lacking `.config` to exercise both guard clauses (~0.8 pp).
- `experiments/bundestag_qwen15b_lora_mps/preparer.py` leaves `OSError` handling untested.【F:.ldres/coverage-opportunities.md†L68-L73】 Use a tmp directory with revoked permissions to trigger the failure and confirm the fallback (~1.2 pp).
- Protocol-backed tokenizer/logging helpers need direct tests through their concrete implementations to guarantee all protocol-required branches execute (~0.3–0.5 pp).【F:.ldres/coverage-opportunities.md†L95-L108】

## Milestone checkpoints

1. **Short term (next PRs)** – Deliver the CLI and tokenizer suites. Expect branch coverage to climb into the mid-80s (≈ +9–10 pp) and justify raising the branch `--fail-under` gate toward 90, matching the roadmap milestone for deterministic modules.【F:docs/coverage/roadmap.md†L40-L44】
1. **Mid term** – Land runtime seams and checkpointing tests, bringing branch coverage into the low 90s and enabling the roadmap’s second milestone focused on protocol interfaces and tighter gates.【F:docs/coverage/roadmap.md†L45-L55】
1. **Final push** – Close out inference, experiment prep, and protocol gaps to reach ≥98%, then decide whether the remaining hardware-dependent CUDA branch warrants a documented skip under the “impossible-to-test” clause or an emulation strategy before locking the gate at 100%.【F:docs/coverage/roadmap.md†L51-L61】【F:.dev-guidelines/TESTING.md†L203-L218】

## Execution playbook and verification

- **Implementation steps**
  - Build new CLI coverage via `pytest` and Typer’s `CliRunner`, keeping all command invocations deterministic by using tmp directories and injectable callables instead of monkeypatching.【F:src/ml_playground/cli.py†L137-L441】【F:.dev-guidelines/TESTING.md†L155-L177】
  - Extend tokenizer tests with parametrization/property data to capture guarded metadata branches while staying within the documented testing surface.【F:src/ml_playground/data_pipeline/transforms/tokenization.py†L39-L104】
  - Introduce dependency-injection seams for runtime CUDA helpers so tests can simulate both success and failure without requiring actual GPUs; mark any true hardware-only checks with `pytest.skipif` plus a rationale in docstrings.【F:src/ml_playground/training/hooks/runtime.py†L33-L51】【F:.dev-guidelines/TESTING.md†L203-L218】
  - Exercise checkpoint cleanup logic through tmp directories with controlled permissions, ensuring filesystem edge cases are covered deterministically.【F:.ldres/coverage-opportunities.md†L31-L48】
  - Cover inference/prep fallbacks by instantiating minimal dataclasses lacking optional attributes and by using permission-locked tmp paths to trigger documented `OSError` branches.【F:.ldres/coverage-opportunities.md†L59-L73】
- **Verification loop**
  - After each feature PR, regenerate metrics with `uvx --from . dev-tasks coverage-report` and confirm that badges and XML outputs are updated before committing.
  - Ratchet `.githooks/.pre-commit-config.yaml` thresholds once branch coverage clears each milestone (85 → 90 → 95) to keep regressions out.
  - Document any intentional skips or hardware limitations directly in the associated tests and cross-link the rationale in `docs/coverage/roadmap.md` to maintain transparency.

Executing these steps keeps us aligned with the ultra-strict testing policy, preserves deterministic runtimes, and provides a clear ladder for ratcheting branch coverage to the mandated 100% ceiling without resorting to disallowed mocking techniques.【F:.dev-guidelines/TESTING.md†L203-L218】【F:.dev-guidelines/TESTING.md†L155-L177】
