from __future__ import annotations


import pytest
from pytest_mock import MockerFixture

from ml_playground.cli import main


def test_sample_routes_to_injected_sampler(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """CLI should dispatch 'sample speakger' to the unified injected-config sampler path (_run_sample)."""
    # Patch the unified run path; we don't want to import heavy experiment deps
    run_sample = mocker.patch("ml_playground.cli._run_sample")

    # Also ensure legacy entrypoint is NOT consulted anymore
    called = {"count": 0}

    def _fake_sample_from_toml(path):  # type: ignore[no-untyped-def]
        called["count"] += 1

    monkeypatch.setattr(
        "ml_playground.experiments.speakger.sampler.sample_from_toml",
        _fake_sample_from_toml,
        raising=False,
    )

    # Act: run CLI with experiment auto-resolved config (no SystemExit on success)
    main(["sample", "speakger"])

    # Assert: unified path called once; legacy path not invoked
    run_sample.assert_called_once()
    assert called["count"] == 0
