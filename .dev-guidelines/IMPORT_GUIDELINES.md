---
trigger: always_on
description: Import standards enforcing PEP 420 namespaces and strict module boundaries
---

# Import Guidelines: PEP 420 Namespace Policy

<details>
<summary>Related documentation</summary>

- [Developer Guidelines Index](./Readme.md) – Entry point for core policies and quick-start commands.
- [Development Practices](./DEVELOPMENT.md) – Commit standards, tooling policies, and architecture notes.

</details>

## Table of Contents

- [Core Principles](#core-principles)
- [PEP 420 Namespace Defaults](#pep-420-namespace-defaults)
- [Mandatory Rules (No Exceptions Without Approval)](#mandatory-rules-no-exceptions-without-approval)
- [Canonical Patterns](#canonical-patterns)
- [Exception Handling Checklist](#exception-handling-checklist)
- [Tooling Enforcement](#tooling-enforcement)
- [Review Checklist](#review-checklist)
- [Rationale](#rationale)

## Core Principles

- **Single, canonical source**: import symbols from the concrete module where they are defined.
- **Namespace transparency**: rely on PEP 420 to compose packages; avoid artificial facades or re-exports.
- **Predictable imports**: keep imports cheap, deterministic, and side-effect free.

## PEP 420 Namespace Defaults

- Every `ml_playground/`, `tests/`, and `tools/` sub-package is an implicit namespace by default.
- Do **not** add `__init__.py` merely for package recognition; Python already discovers namespaces.
- Directory layout alone defines the public API surface. Avoid hidden shims that obscure import paths.

## Mandatory Rules (No Exceptions Without Approval)

1. **Absolute, module-level imports only**

   - Use absolute paths rooted at the repository package (e.g., `from ml_playground.core.foo import Bar`).
   - Prohibited: relative imports, umbrella modules, or implicit re-exports.

1. **No star imports**

   - `from module import *` is disallowed in all code and tests.

1. **Import placement**

   - Place imports at module top level, beneath the docstring.
   - Exceptions: documented cycle breaks or optional dependencies (see rules 8–9).

1. **Ordering and grouping**

   - Maintain three groups separated by single blank lines:
     1. Standard library
     1. Third-party
     1. First-party (`ml_playground`, `tests`, `tools`)
   - Alphabetize within each group. Allow the formatter/import organizer to enforce ordering.

1. **Aliasing**

   - Only the conventional aliases `import numpy as np` and `import pandas as pd` are allowed.
   - Otherwise, import the module without aliases.

1. **Re-exports and facades**

   - Prohibited: exposing symbols via shims, compatibility modules, or implicit public APIs.
   - Consumers must import directly from the module that defines the symbol.

1. **Side-effect free imports**

   - Importing a module must not trigger I/O, logging setup, network calls, or global mutations.
   - If unavoidable, relocate the side effect behind an explicit function or command and document the rationale with a
     nearby TODO comment following the repository format (`# TODO Remove <context>: <reason>`).

1. **Optional dependencies**

   - Import optional packages in the narrowest scope that needs them.
   - On missing dependency, raise a clear error with installation guidance.
   - Prohibited: optional dependencies at module top level.

1. **Cycle handling**

   - Preferred fix: refactor to remove the cycle.
   - Temporary escape hatch: a documented local import (`# TODO Remove cycle break: <reason>`) inside the dependent
     function.
     Track a follow-up task to remove the cycle.

1. **Type-only imports**

   - Use `typing.TYPE_CHECKING` for heavy or optional typing dependencies.
   - Prefer postponed annotations (enabled via `from __future__ import annotations` or Python ≥3.13 defaults).

1. **Lazy imports**

   - Disallowed unless both apply: the import breaks a hard cycle or meaningfully reduces cold-start cost.
   - Document with `# TODO Remove lazy import: <reason + impact>` and open a task to measure/remove when feasible.

1. **`__init__.py` exception policy**

   - Default: omit `__init__.py` entirely.
   - Allowed only when the file contains one of the following, and the rationale is documented near the file:
     - Metadata required by packaging (`__version__`, entry-point registration) pending migration.
     - Third-party compatibility that cannot yet consume namespace packages.
     - Explicit plugin registration that cannot be moved behind a callable without breaking integration.
   - Exception files must stay side-effect free, include a TODO comment describing the exception, and must not
     re-export symbols.

## Canonical Patterns

### Import specific names from a concrete module

```python
from ml_playground.core.optimizer import Optimizer
```

### Import a module wholesale

```python
import ml_playground.data_pipeline.sources.text as text_sources
```

### Optional dependency (function scoped)

```python
def export_to_parquet(df, path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export requires 'pyarrow'. Install it to use this feature.") from exc
    pq.write_table(df, path)
```

### Type-only import for heavy dependencies

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_playground.models.core.model import GPT
```

### Documented temporary cycle break

```python
def compute_metrics():
    # Cycle break: trainer.metrics imports compute_metrics during warm start
    from ml_playground.training.metrics import aggregate

    return aggregate()
```

## Exception Handling Checklist

- Document every surviving `__init__.py` with a `README` snippet or in-line comment that states the reason.
- Open a tracking task for each exception to ensure eventual migration to pure namespaces.
- Ensure no exception file performs imports that could create hidden re-export paths.

## Tooling Enforcement

- Run `uv run ci-tasks quality` (ruff + formatter) before every commit to enforce import style automatically.
- Keep type checkers (`pyright`, `mypy`) green; use `TYPE_CHECKING` guards to avoid runtime imports.
- Pair import-structure changes with relevant tests/documentation updates.

## Review Checklist

- **Namespace**: Does the change avoid introducing new `__init__.py` files or document approved exceptions?
- **Imports**: Are all imports absolute and taken from canonical modules?
- **Ordering**: Do import blocks follow stdlib / third-party / first-party grouping and alphabetical order?
- **Side effects**: Does the diff introduce any top-level work beyond definitions? If so, is it justified?
- **Cycles/optional deps**: Are local imports properly documented with follow-up tasks?

## Rationale

- PEP 420 namespaces eliminate boilerplate and make plugin/extensibility work straightforward.
- Explicit imports improve searchability, refactoring safety, and API stability.
- Deterministic, side-effect free imports yield faster startup time and more reliable tests.
- Reduced choice limits bikeshedding in reviews and keeps the codebase consistent.

Adhering to these rules keeps the namespace flat, predictable, and ready for modular expansion.
