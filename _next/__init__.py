"""
_next: Strict, typed, single-entry module for preparing data, training, and sampling.

Usage examples:
  python -m _next.cli prepare shakespeare
  python -m _next.cli train _next/configs/shakespeare_cpu.toml
  python -m _next.cli sample _next/configs/shakespeare_cpu.toml
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
