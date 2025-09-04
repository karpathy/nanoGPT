"""
Compatibility shim for the optional 'lit_nlp' package.

Purpose:
- Allow local development to use a real lit-nlp installation if present, even
  though this repository contains a top-level 'lit_nlp' directory.
- If lit-nlp is not installed, importing 'lit_nlp' should fail with a clear
  message, prompting the user to install the optional dependency.

How it works:
- We locate the installed distribution 'lit-nlp' via importlib.metadata.
- If found, we set this package's __path__ to point at the real package's
  directory so that submodule imports like 'lit_nlp.api' resolve from the real
  implementation. We also execute the real __init__ to populate attributes.
- If not found, we raise ImportError with guidance.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
import importlib.util
from pathlib import Path

try:
    dist = distribution("lit-nlp")
except PackageNotFoundError as e:  # No real lit-nlp installed
    raise ModuleNotFoundError(
        "lit_nlp is not installed. Install with 'uv sync --extra lit' or 'uv add lit-nlp'. "
        "See docs/LIT.md for details."
    ) from e

# Find the real package root by locating lit_nlp/__init__.py in the distribution files
pkg_root: Path | None = None
for f in dist.files or ():  # type: ignore[arg-type]
    if str(f).endswith("lit_nlp/__init__.py"):
        pkg_root = Path(str(dist.locate_file(f))).parent
        break
if pkg_root is None:
    # Fallback: try common site-packages location based on distribution location
    loc = Path(str(dist.locate_file("."))).resolve()
    candidate = loc / "lit_nlp"
    if (candidate / "__init__.py").exists():
        pkg_root = candidate

if pkg_root is None:
    raise ImportError(
        "Failed to locate installed 'lit_nlp' package files; ensure lit-nlp is properly installed."
    )

# Configure this shim package as a namespace that points to the real implementation
__path__ = [str(pkg_root)]  # type: ignore[assignment]

# Execute the real __init__.py to populate attributes on this module
_real_init = pkg_root / "__init__.py"
spec = importlib.util.spec_from_file_location("lit_nlp.__init__", _real_init)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load real 'lit_nlp' package initialisation module")
_real_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_real_mod)  # type: ignore[arg-type]

# Re-export public attributes from the real module
for name in dir(_real_mod):
    if name.startswith("__") and name not in {"__all__", "__version__"}:
        continue
    globals()[name] = getattr(_real_mod, name)
