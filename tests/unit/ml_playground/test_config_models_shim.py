from __future__ import annotations

# Simply import and touch a few attributes to cover the shim
from ml_playground import config_models as shim


def test_config_models_shim_exports():
    # Ensure the module exposes expected names
    assert hasattr(shim, "TrainerConfig")
    assert hasattr(shim, "SamplerConfig")
    assert hasattr(shim, "DataConfig")
    assert hasattr(shim, "RuntimeConfig")
    assert hasattr(shim, "load_toml")
