"""Auto-mark all tests in tests/integration/ as integration tests.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration
