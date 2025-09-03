from __future__ import annotations
from types import ModuleType
from unittest.mock import patch

from ml_playground.analysis import lit_integration as li


def test_lit_integration_server_path(monkeypatch):
    # Construct a fake 'lit_nlp' package hierarchy to avoid triggering local shim
    lit_pkg = ModuleType("lit_nlp")
    api_pkg = ModuleType("lit_nlp.api")
    dataset_mod = ModuleType("lit_nlp.api.dataset")
    model_mod = ModuleType("lit_nlp.api.model")
    types_mod = ModuleType("lit_nlp.api.types")

    # Minimal classes/constructs expected by code
    dataset_mod.Dataset = object  # type: ignore[attr-defined]
    model_mod.Model = object  # type: ignore[attr-defined]
    types_mod.TextSegment = lambda *a, **k: ("TextSegment", a, k)  # type: ignore[attr-defined]

    class FakeServer:
        def __init__(self, models, datasets):
            self.models = models
            self.datasets = datasets

    served = {}

    def fake_serve(app, port=None, host=None, open_browser=None):
        served["called"] = True
        served["port"] = port
        served["host"] = host
        served["open_browser"] = open_browser

    server_mod = ModuleType("lit_nlp.server")
    server_mod.Server = FakeServer  # type: ignore[attr-defined]
    server_mod.serve = fake_serve  # type: ignore[attr-defined]

    # Register into sys.modules before the function under test imports them
    with patch.dict(
        "sys.modules",
        {
            "lit_nlp": lit_pkg,
            "lit_nlp.api": api_pkg,
            "lit_nlp.api.dataset": dataset_mod,
            "lit_nlp.api.model": model_mod,
            "lit_nlp.api.types": types_mod,
            "lit_nlp.server": server_mod,
        },
    ):
        li.run_server_bundestag_char(host="0.0.0.0", port=1234, open_browser=False)
        assert served.get("called") is True
        assert served["port"] == 1234
        assert served["host"] == "0.0.0.0"
        assert served["open_browser"] is False
