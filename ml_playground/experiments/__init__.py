from __future__ import annotations
from typing import Callable, Dict
from importlib import import_module
from importlib import resources

# Registry for experiment-level dataset preparers (mirrors previous datasets.PREPARERS)
PREPARERS: Dict[str, Callable[[], None]] = {}


def load_preparers() -> None:
    """Plugin loader: import experiment preparers to populate PREPARERS.

    Strict mode: only class-based API is supported. An experiment must expose
    a preparer.py with a class that has a .prepare method. A zero-arg callable
    is registered that instantiates the class and calls .prepare(PreparerConfig()).
    """
    if PREPARERS:
        # Already populated (or tests monkeypatched it)
        return

    pkg = "ml_playground.experiments"
    try:
        root = resources.files(pkg)
    except Exception:
        # If discovery fails (e.g., frozen environments), do nothing; callers may
        # have injected PREPARERS via tests or alternative mechanisms.
        return

    for entry in root.iterdir():
        try:
            if not entry.is_dir():
                continue
            exp_name = entry.name

            # Strict API: preparer.py with a class exposing .prepare
            prep_file = entry / "preparer.py"
            if not prep_file.is_file():
                continue
            try:
                mod = import_module(f"{pkg}.{exp_name}.preparer")
                # Find first class with a 'prepare' attribute
                cls = None
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "prepare"):
                        cls = attr
                        break
                if cls is None:
                    continue
                from ml_playground.prepare import PreparerConfig  # local import

                def _make_fn(_cls=cls) -> None:  # type: ignore[no-redef]
                    inst = _cls()
                    try:
                        inst.prepare(PreparerConfig())  # type: ignore[attr-defined]
                    except TypeError:
                        # Some preparers may accept no args
                        inst.prepare()  # type: ignore[call-arg, attr-defined]

                PREPARERS.setdefault(exp_name, _make_fn)
            except Exception as e:
                raise SystemExit(f"Failed to load experiment '{exp_name}': {e}")
        except Exception:
            # Defensive: ignore any unexpected filesystem/resource issues per entry
            continue
