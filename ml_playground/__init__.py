"""
ml_playground: Strict, typed, single-entry module for preparing data, training, and sampling. (formerly `_next`)

Usage examples:
  python -m ml_playground.cli prepare shakespeare
  python -m ml_playground.cli train ml_playground/experiments/shakespeare/shakespeare_cpu.toml
  python -m ml_playground.cli sample ml_playground/experiments/shakespeare/shakespeare_cpu.toml
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
