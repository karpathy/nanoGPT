from __future__ import annotations

# Compatibility shim retained for import stability.
# Delegates to the single source in ml_playground.config.

from .config import AppConfig, load_toml as load_config

__all__ = ["load_config", "AppConfig"]
