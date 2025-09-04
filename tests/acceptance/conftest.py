"""Auto-mark all tests in tests/acceptance/ as acceptance tests.

This keeps suite selection easy via `-m acceptance` and allows
separate reporting/CI handling for acceptance tests.
"""
from __future__ import annotations

import pytest

# Apply the 'acceptance' marker to every test in this package
pytestmark = pytest.mark.acceptance
