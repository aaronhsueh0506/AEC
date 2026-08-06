"""Make the adjacent standalone Python AEC implementation importable in tests."""

from __future__ import annotations

import os
import sys


PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
