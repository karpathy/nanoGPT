from __future__ import annotations

"""
Analysis helpers and integrations for ml_playground.

Currently provides a lightweight wrapper to launch the optional LIT UI for
interactive inspection as a proof of concept focused on the bundestag_char
experiment.

This module is optional and only used when invoking the CLI command
`python -m ml_playground.cli analyze bundestag_char`.
"""

__all__ = [
    "run_server_bundestag_char",
]

# Re-export the entrypoint for convenience
from .lit_integration import run_server_bundestag_char  # noqa: E402
