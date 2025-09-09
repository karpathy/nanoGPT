"""Shared test configuration and fixtures for ml_playground tests.

This module provides session-level fixtures and configuration that applies
to all tests in the ml_playground test suite.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Callable
import random
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def _seed_randomness() -> None:
    """Seed random number generators for deterministic test runs.

    This fixture automatically runs once per test session to ensure
    reproducible results across all tests that use randomness.
    """
    random.seed(1337)
    np.random.seed(1337)


# ----------------------------------------------------------------------------
# Global path fixture(s)
# ----------------------------------------------------------------------------


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """Provide a conventionally named output directory under tmp_path.

    Many tests construct out_dir = tmp_path / "out"; centralize this for reuse.
    """
    p = tmp_path / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Central TOML builders for common experiment shapes
# ----------------------------------------------------------------------------


def _fmt_path(p: Path) -> str:
    # Ensure forward slashes in TOML strings for portability
    return str(p.as_posix())


def minimal_full_experiment_toml(
    dataset_dir: Path,
    out_dir: Path,
    *,
    extra_optim: str = "",
    extra_train: str = "",
    extra_sample: str = "",
    extra_sample_sample: str = "",
    include_train_data: bool = True,
    include_train_runtime: bool = True,
    include_sample: bool = True,
) -> str:
    """Return a minimal, strict ExperimentConfig TOML with overridable sections.

    Parameters allow injecting extra lines per section via string snippets
    (already properly indented TOML lines).
    """
    base = f"""
    [prepare]

    [train.model]
    """
    if include_train_data:
        base += f"""
        [train.data]
        dataset_dir = "{_fmt_path(dataset_dir)}"
        """
    base += f"""
    [train.optim]
    {extra_optim}

    [train.schedule]
    """
    if include_train_runtime:
        base += f"""
        [train.runtime]
        out_dir = "{_fmt_path(out_dir)}"
        {extra_train}
        """
    if include_sample:
        base += f"""
        [sample]
        [sample.runtime]
        out_dir = "{_fmt_path(out_dir)}"
        {extra_sample}
        [sample.sample]
        {extra_sample_sample}
        """
    return dedent(base)


@pytest.fixture()
def toml_minimal_factory() -> Callable[[Path, Path], str]:
    """Factory returning a minimal full ExperimentConfig TOML string.

    Usage:
        text = toml_minimal_factory(dataset_dir, out_dir)
    For overrides, call minimal_full_experiment_toml directly if needed.
    """
    def _factory(dataset_dir: Path, out_dir: Path) -> str:
        return minimal_full_experiment_toml(dataset_dir, out_dir)

    return _factory
