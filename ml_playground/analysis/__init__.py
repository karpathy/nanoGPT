"""
Analysis helpers and integrations for ml_playground.

Optional: a minimal LIT UI integration focused on the bundestag_char PoC.
"""

from __future__ import annotations

from ml_playground.analysis.lit.integration import run_server_bundestag_char  # noqa: F401

__all__ = [
    "run_server_bundestag_char",
]

# Re-export the LIT integration entrypoint for convenience
