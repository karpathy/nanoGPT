import pytest

pytest.importorskip(
    "lit_nlp",
    reason="tv: Feature work in progress. lit-nlp dev dependency not compatible with Python 3.13.5",
)


def test_run_server_bundestag_char_monkeypatched_serve(monkeypatch):
    # Import inside test to ensure optional dependency errors surface properly
    from ml_playground.analysis.lit_integration import run_server_bundestag_char

    # Monkeypatch lit_nlp.server.serve to avoid blocking
    try:
        import lit_nlp.server as lit_server
    except Exception as e:  # pragma: no cover - guarded by importorskip
        pytest.skip(str(e))
        return

    calls = {"serve": 0}

    def _fake_serve(app, *args, **kwargs):  # noqa: ANN001 - external signature
        calls["serve"] += 1
        return None

    monkeypatch.setattr(lit_server, "serve", _fake_serve, raising=True)

    # Execute the server function; with patched serve it should return immediately
    run_server_bundestag_char(host="127.0.0.1", port=0, open_browser=False)

    assert calls["serve"] == 1
