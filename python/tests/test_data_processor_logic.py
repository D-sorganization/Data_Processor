import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Try to import process_single_csv_file
try:
    from Data_Processor_r0 import process_single_csv_file
except ImportError:
    # If not found (maybe pytest collection issue), try relative import if possible or fail
    try:
        from python.data_processor.Data_Processor_r0 import process_single_csv_file
    except ImportError:
        # Last resort: add path explicitly here if conftest didn't work as expected
        sys.path.append(str(Path(__file__).parent.parent / "data_processor"))
        from Data_Processor_r0 import process_single_csv_file


@pytest.fixture
def sample_csv(tmp_path) -> str:
    """Create a sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "Time": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
            "Signal1": np.random.rand(100),
            "Signal2": np.random.rand(100),
        }
    )
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_process_single_csv_file_basic(sample_csv) -> None:
    """Test basic processing of a single CSV file."""
    settings = {
        "selected_signals": ["Signal1"],
        "filter_type": "None",
        "resample_enabled": False,
    }

    result = process_single_csv_file(sample_csv, settings)

    assert result is not None
    assert "Signal1" in result.columns
    assert "Signal2" not in result.columns
    assert len(result) == 100


def test_process_single_csv_file_with_filtering(sample_csv) -> None:
    """Test processing with filtering enabled."""
    settings = {
        "selected_signals": ["Signal1"],
        "filter_type": "Moving Average",
        "ma_window": 5,
        "resample_enabled": False,
    }

    result = process_single_csv_file(sample_csv, settings)

    assert result is not None
    assert "Signal1" in result.columns
    assert not result.empty


def test_process_single_csv_file_with_resampling(sample_csv) -> None:
    """Test processing with resampling enabled."""
    settings = {
        "selected_signals": ["Signal1"],
        "filter_type": "None",
        "resample_enabled": True,
        "resample_rule": "10s",
    }

    result = process_single_csv_file(sample_csv, settings)

    assert result is not None
    # 100 seconds / 10s = 10 samples (or 11 depending on inclusion)
    assert len(result) in [10, 11]


def test_process_single_csv_file_invalid_file() -> None:
    """Test processing with an invalid file path."""
    settings = {
        "selected_signals": ["Signal1"],
    }
    result = process_single_csv_file("non_existent_file.csv", settings)
    assert result is None

