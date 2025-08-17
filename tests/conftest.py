"""Shared test configuration and fixtures for ml_playground tests.

This module provides session-level fixtures and configuration that applies
to all tests in the ml_playground test suite.
"""

from __future__ import annotations

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
