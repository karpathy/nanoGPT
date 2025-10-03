from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Dict

import pytest

from ml_playground.analysis.lit.integration import run_server_bundestag_char


@pytest.mark.integration
def test_lit_integration_raises_when_lit_missing():
    with pytest.raises(RuntimeError) as ei:
        run_server_bundestag_char(host="127.0.0.1", port=0, open_browser=False)
    msg = str(ei.value)
    assert "LIT is not available" in msg
    # Provide a hint that the suggested command appears in the message
    assert re.search(r"uv\s+sync\s+--extra\s+lit|uv\s+add\s+lit-nlp", msg)


def test_lit_integration_server_path(tmp_path: Path) -> None:
    # Build a minimal on-disk lit_nlp package to satisfy imports without monkeypatching
    pkg_root = tmp_path / "fake_lit"
    lit_root = pkg_root / "lit_nlp"
    api_root = lit_root / "api"

    api_root.mkdir(parents=True, exist_ok=True)

    (lit_root / "__init__.py").write_text(
        "__version__ = '0.0-test'\n", encoding="utf-8"
    )
    (api_root / "__init__.py").write_text("", encoding="utf-8")
    (api_root / "dataset.py").write_text(
        "class Dataset:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    (api_root / "model.py").write_text(
        "class Model:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    (api_root / "types.py").write_text(
        "class TextSegment:\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return ('TextSegment', args, kwargs)\n",
        encoding="utf-8",
    )

    server_state: Dict[str, object] = {}
    (lit_root / "server.py").write_text(
        "SERVER_STATE = {}\n"
        "class Server:\n"
        "    def __init__(self, models, datasets):\n"
        "        SERVER_STATE['models'] = models\n"
        "        SERVER_STATE['datasets'] = datasets\n"
        "        self.models = models\n"
        "        self.datasets = datasets\n"
        "\n"
        "def serve(app, port=None, host=None, open_browser=None):\n"
        "    SERVER_STATE['called'] = True\n"
        "    SERVER_STATE['port'] = port\n"
        "    SERVER_STATE['host'] = host\n"
        "    SERVER_STATE['open_browser'] = open_browser\n"
        "    SERVER_STATE['app'] = app\n",
        encoding="utf-8",
    )

    importlib.invalidate_caches()

    module_names = [
        "lit_nlp",
        "lit_nlp.api",
        "lit_nlp.api.dataset",
        "lit_nlp.api.model",
        "lit_nlp.api.types",
        "lit_nlp.server",
    ]
    preserved = {name: sys.modules.pop(name, None) for name in module_names}

    sys.path.insert(0, str(pkg_root))
    try:
        run_server_bundestag_char(host="0.0.0.0", port=1234, open_browser=False)
        server_mod = importlib.import_module("lit_nlp.server")
        server_state = getattr(server_mod, "SERVER_STATE")
        assert server_state.get("called") is True
        assert server_state.get("port") == 1234
        assert server_state.get("host") == "0.0.0.0"
        assert server_state.get("open_browser") is False
    finally:
        sys.path.remove(str(pkg_root))
        for name in module_names:
            sys.modules.pop(name, None)
            preserved_mod = preserved[name]
            if preserved_mod is not None:
                sys.modules[name] = preserved_mod
