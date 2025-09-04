"""Auto-mark all tests in tests/e2e/ as end-to-end tests.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e
