from __future__ import annotations

import pytest

# CodecManager has been removed from production code. Keep file for historical
# context but skip all tests.
pytestmark = pytest.mark.skip(reason="CodecManager removed from production code")
