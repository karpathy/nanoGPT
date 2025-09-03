from __future__ import annotations
import re
import pytest

from ml_playground.analysis.lit_integration import run_server_bundestag_char


@pytest.mark.integration
def test_lit_integration_raises_when_lit_missing():
    with pytest.raises(RuntimeError) as ei:
        # LIT optional dependency is not installed in CI; function should raise with guidance.
        run_server_bundestag_char(host="127.0.0.1", port=0, open_browser=False)
    msg = str(ei.value)
    assert "LIT is not available" in msg
    # Provide a hint that the suggested command appears in the message
    assert re.search(r"uv\s+sync\s+--extra\s+lit|uv\s+add\s+lit-nlp", msg)
