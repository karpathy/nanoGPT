from __future__ import annotations

import pytest

# Codec creation helpers have been removed from production code.
pytestmark = pytest.mark.skip(reason="Codec helpers removed from production code")
