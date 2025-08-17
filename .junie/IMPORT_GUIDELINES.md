# Import Guidelines: Strict Policy and Rationale

This is a prescriptive, low‑choice standard for all Python imports in the codebase. Follow as written.

## Core Principles
- Single source of truth: import only from the definitive submodule that defines the symbol.
- Zero ambiguity: one allowed way for each common scenario.
- No hidden behavior: imports must be pure, cheap, and deterministic.

## Mandatory Rules (No Exceptions Without Approval)

1. **Absolute, submodule-level imports only**
   - Use absolute project-rooted paths.
   - Import directly from the concrete submodule that defines the symbol.
   - Prohibited: relative imports, umbrella/facade imports, and re-exports.

2. **No star imports**
   - Prohibited: from module import *.

3. **Import location**
   - All imports at top-of-file, below the module docstring.
   - Prohibited: local/function-scope imports, except for approved cycle breaks or optional deps (see rules 8–9).

4. **Ordering and grouping**
   - Exactly three groups, in order, one blank line between groups:
     1) Standard library
     2) Third-party
     3) Local/project
   - Alphabetize within each group.
   - Use the project's formatter/import organizer to enforce this automatically.

5. **Aliasing**
   - Allowed only for the following well-known libraries:
     - import numpy as np
     - import pandas as pd
   - Otherwise, no aliases. Use full module paths.

6. **Re-exports and facades**
   - Prohibited: exposing symbols via package-level __init__.py, "compat" modules, or import shims.
   - Consumers must import from the canonical submodule.

7. **Side-effect free imports**
   - Importing must not perform I/O, network calls, logging configuration, global registrations, or mutate global state.
   - If unavoidable for a plugin mechanism, move side effects behind an explicit function that callers invoke manually.

8. **Optional dependencies**
   - Import optional packages inside the narrowest function that needs them.
   - On missing dependency, raise a clear error with installation guidance.
   - Prohibited: optional deps at module top-level.

9. **Cycle handling**
   - First choice: refactor to remove the cycle (extract shared code to a lower-level module).
   - If refactor is not immediately feasible, a single, narrowly-scoped local import is permitted inside the function that needs it, with a code comment "Cycle break: <short rationale>". Track a task to remove the cycle.

10. **Type-only imports**
    - Use typing.TYPE_CHECKING guards for heavy or optional typing dependencies.
    - Prefer postponed evaluation of annotations (default in modern Python) to avoid runtime import costs.

11. **Lazy imports**
    - Not allowed by default.
    - Allowed only when both conditions hold: breaks a hard import cycle or defers a large, cold-path dependency with measurable startup benefit. Must be documented with a comment "Lazy import: <reason + expected impact>".

12. **__init__.py usage**
    - May exist for package recognition or minimal metadata only.
    - Prohibited: symbol re-exports, wildcard exports, or public API surfaces.

## Canonical Patterns

### Import few names from a concrete submodule:
```python
from project.module.submodule import Foo, Bar
```

### Import many names from one submodule:
```python
import project.module.submodule as submodule  # avoid custom aliases
```

### Optional dependency (function-scoped):
```python
def export_to_parquet(df, path):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("Parquet export requires 'pyarrow'. Install it to use this feature.") from e
    pq.write_table(df, path)
```

### Type-only import for a heavy/optional type:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heavy_lib import HugeType
```

### Temporary cycle break (documented):
```python
def compute():
    # Cycle break: runtime imports runner, runner imports compute
    from project.core.runner import run
    return run()
```

## Tooling Enforcement

- Run the project's linter/formatter/import organizer before every commit to enforce ordering and grouping.
- Type checkers must pass with type-only guards for heavy/optional types.

## Review Checklist (Must Pass All)

- Are all imports absolute and from definitive submodules?
- Any star imports, umbrella paths, or re-exports? (Must be none.)
- Import groups in order: stdlib, third-party, local? Alphabetized?
- Any top-level side effects or logging config triggered by import?
- Any function-scope imports? If yes, only for optional deps or documented cycle break?
- Optional deps imported only where needed with a clear error message?
- Heavy/optional type imports guarded with TYPE_CHECKING?

## Rationale (Concise)

- Explicitness improves searchability, refactoring safety, and API stability.
- Deterministic, side-effect-free imports yield faster startup and more reliable tests.
- Reduced choice minimizes bikeshedding and accelerates reviews.
- Tight rules prevent accidental public APIs and long-lived cycles.

Adhering to these rules ensures clarity, stability, and predictable behavior across the codebase.