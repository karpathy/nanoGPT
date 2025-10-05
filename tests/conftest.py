"""Shared test configuration and fixtures for ml_playground tests.

This module provides session-level fixtures and configuration that applies
to all tests in the ml_playground test suite.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterator
import random
import numpy as np
import pytest
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase

from ml_playground.configuration.models import SharedConfig


@pytest.fixture(autouse=True, scope="session")
def _seed_randomness() -> None:
    """Seed random number generators for deterministic test runs.

    This fixture automatically runs once per test session to ensure
    reproducible results across all tests that use randomness.
    """
    random.seed(1337)
    np.random.seed(1337)


# ----------------------------------------------------------------------------
# Hypothesis global database location
# ----------------------------------------------------------------------------

# Ensure Hypothesis stores its example database under the centralized cache.
# This applies regardless of how pytest is invoked (Makefile, pre-commit, IDE, etc.).
settings.register_profile(
    "repo-default",
    database=DirectoryBasedExampleDatabase(Path(".cache/hypothesis")),
)
settings.load_profile("repo-default")


def pytest_load_initial_conftests(args, early_config, parser) -> None:  # type: ignore[override]
    """Early hook: ensure no top-level .hypothesis dir exists before collection.

    Pre-commit runs pytest with -W error; Hypothesis plugin warns when it sees
    a top-level .hypothesis dir skipped by norecursedirs. We remove it here to
    avoid the warning entirely.
    """
    top = Path.cwd() / ".hypothesis"
    try:
        if top.exists():
            # Safety: only remove if it's a directory inside repo root
            if top.is_dir():
                for p in sorted(top.rglob("*"), reverse=True):
                    try:
                        if p.is_file() or p.is_symlink():
                            p.unlink(missing_ok=True)
                        elif p.is_dir():
                            p.rmdir()
                    except Exception:
                        pass
                top.rmdir()
    except Exception:
        # Non-fatal: better to proceed than fail early
        pass


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
# Shared helpers for CLI/config property and unit tests
# ----------------------------------------------------------------------------


@pytest.fixture()
def shared_config_factory() -> Callable[[Path], SharedConfig]:
    """Return a factory that builds a `SharedConfig` rooted at the provided path."""

    def _factory(base_dir: Path) -> SharedConfig:
        dataset_dir = base_dir / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        train_dir = base_dir / "train"
        train_dir.mkdir(exist_ok=True)
        sample_dir = base_dir / "sample"
        sample_dir.mkdir(exist_ok=True)
        config_path = base_dir / "config.toml"
        config_path.write_text("{}", encoding="utf-8")
        return SharedConfig(
            experiment="demo",
            config_path=config_path,
            project_home=base_dir,
            dataset_dir=dataset_dir,
            train_out_dir=train_dir,
            sample_out_dir=sample_dir,
        )

    return _factory


@pytest.fixture()
def override_attr() -> Callable[[object, str, object], Iterator[None]]:
    """Provide a context manager for temporarily overriding attributes on objects."""

    @contextmanager
    def _override(target: object, attr: str, value: object) -> Iterator[None]:
        original = getattr(target, attr)
        setattr(target, attr, value)
        try:
            yield
        finally:
            setattr(target, attr, original)

    return _override


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
    base = """
    [prepare]

    [train.model]
    """
    if include_train_data:
        base += """
        [train.data]
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
    # Add shared section: tied to provided dataset_dir/out_dir; generic experiment metadata
    base += f"""
    [shared]
    experiment = "exp"
    config_path = "{_fmt_path(out_dir.parent / "cfg.toml")}"
    project_home = "{_fmt_path(out_dir.parent)}"
    dataset_dir = "{_fmt_path(dataset_dir)}"
    train_out_dir = "{_fmt_path(out_dir)}"
    sample_out_dir = "{_fmt_path(out_dir)}"
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
