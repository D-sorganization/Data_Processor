"""Pytest configuration for tests.

This file sets up the test environment, including PYTHONPATH configuration
to ensure imports work correctly in both local and CI environments.
"""

import sys
from pathlib import Path

# Add python/src to PYTHONPATH for test imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add python/data_processor to PYTHONPATH for test imports
data_processor_path = Path(__file__).parent.parent / "data_processor"
if str(data_processor_path) not in sys.path:
    sys.path.insert(0, str(data_processor_path))
