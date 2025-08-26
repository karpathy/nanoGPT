from __future__ import annotations
from typing import Callable, Dict
from importlib import import_module
from importlib import resources

# Registry for experiment-level dataset preparers (mirrors previous datasets.PREPARERS)
PREPARERS: Dict[str, Callable[[], None]] = {}


def register(name: str):
    def _wrap(fn: Callable[[], None]) -> Callable[[], None]:
        PREPARERS[name] = fn
        return fn

    return _wrap


def load_preparers() -> None:
    """Plugin loader: import experiment prepare modules to populate PREPARERS.

    Dynamically discovers subpackages under ml_playground.experiments and
    imports their 'prepare' module if present. This avoids hardcoding any
    experiment names in general code and keeps all experiment-specific logic
    confined to the experiments/ subpackages.
    """
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
            # Only consider experiment packages that provide a prepare.py
            if not (entry / "prepare.py").is_file():
                continue
            mod_name = f"{pkg}.{entry.name}.prepare"
            try:
                import_module(mod_name)
            except ModuleNotFoundError:
                # If prepare module not found despite file existence, skip
                continue
            except Exception as e:
                # Fail fast with a clear error for broken experiment packages
                raise SystemExit(f"Failed to load experiment '{entry.name}': {e}")
        except Exception:
            # Defensive: ignore any unexpected filesystem/resource issues per entry
            continue
