from __future__ import annotations


import pytest
from pytest_mock import MockerFixture

from ml_playground.cli import main


def test_sample_routes_to_speakger_integration(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """CLI should dispatch 'sample speakger' to the SpeakGer integration sampler, not classic sampler."""
    # Patch the integration's sampler entrypoint directly to avoid heavy deps
    called = {"count": 0}

    def _fake_sample_from_toml(path):  # type: ignore[no-untyped-def]
        called["count"] += 1

    monkeypatch.setattr(
        "ml_playground.experiments.speakger.sampler.sample_from_toml",
        _fake_sample_from_toml,
        raising=False,
    )

    # Ensure classic sampler is not used on this path
    classic_sample = mocker.patch("ml_playground.cli.sample")

    # Act: run CLI with experiment auto-resolved config
    main(["sample", "speakger"])

    # Assert: integration sampler called once
    assert called["count"] == 1
    # Classic path must not be used
    classic_sample.assert_not_called()
