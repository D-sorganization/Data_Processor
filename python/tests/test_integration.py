"""Integration tests for Data Processor workflows.

These tests validate end-to-end workflows including:
- Loading CSV files
- Detecting signals
- Applying filters
- Integration and differentiation
- Saving results

Run with: pytest test_integration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path so we can import data_processor package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)


class TestDataLoaderIntegration:
    """Integration tests for data loading workflows."""

    @pytest.fixture()
    def sample_csv_file(self, tmp_path: Path) -> Path:
        """Create a sample CSV file for testing."""
        # Generate sample data
        np.random.seed(42)
        n_rows = 1000

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1S"),
                "temperature": 20
                + 5 * np.sin(np.linspace(0, 10, n_rows))
                + np.random.randn(n_rows) * 0.5,
                "pressure": 100
                + 10 * np.cos(np.linspace(0, 10, n_rows))
                + np.random.randn(n_rows) * 1.0,
                "flow_rate": 50 + 5 * np.random.randn(n_rows),
            },
        )

        # Save to CSV
        csv_file = tmp_path / "sample_data.csv"
        df.to_csv(csv_file, index=False)

        return csv_file

    @pytest.fixture()
    def multiple_csv_files(self, tmp_path: Path) -> list[str]:
        """Create multiple CSV files for testing."""
        np.random.seed(42)
        files = []

        for i in range(3):
            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        f"2024-01-0{i+1}", periods=100, freq="1min",
                    ),
                    f"sensor_{i+1}": np.random.randn(100),
                    "common_signal": np.random.randn(100),
                },
            )

            csv_file = tmp_path / f"data_{i+1}.csv"
            df.to_csv(csv_file, index=False)
            files.append(str(csv_file))

        return files

    def test_load_single_csv(self, sample_csv_file: Path) -> None:
        """Test loading a single CSV file."""
        loader = DataLoader()

        df = loader.load_csv_file(str(sample_csv_file), validate_security=False)

        assert df is not None
        assert len(df) == 1000
        assert "temperature" in df.columns
        assert "pressure" in df.columns
        assert "flow_rate" in df.columns

    def test_detect_signals(self, multiple_csv_files: list[str]) -> None:
        """Test detecting signals from multiple files."""
        loader = DataLoader()

        signals = loader.detect_signals(multiple_csv_files)

        assert "common_signal" in signals
        assert "sensor_1" in signals or "sensor_2" in signals or "sensor_3" in signals
        assert len(signals) >= 4  # timestamp + 3 sensors + common

    def test_load_multiple_files(self, multiple_csv_files: list[str]) -> None:
        """Test loading multiple CSV files."""
        loader = DataLoader()

        dataframes = loader.load_multiple_files(multiple_csv_files)

        assert len(dataframes) == 3
        for df in dataframes.values():
            assert df is not None
            assert len(df) == 100

    def test_detect_time_column(self, sample_csv_file: Path) -> None:
        """Test automatic time column detection."""
        loader = DataLoader()
        df = loader.load_csv_file(str(sample_csv_file), validate_security=False)

        time_col = loader.detect_time_column(df)

        assert time_col == "timestamp"

    def test_convert_time_column(self, sample_csv_file: Path) -> None:
        """Test converting time column to index."""
        loader = DataLoader()
        df = loader.load_csv_file(str(sample_csv_file), validate_security=False)

        time_col = loader.detect_time_column(df)
        df = loader.convert_time_column(df, time_col)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 1000


class TestSignalProcessorIntegration:
    """Integration tests for signal processing workflows."""

    @pytest.fixture()
    def sample_data(self) -> pd.DataFrame:
        """Create sample DataFrame for processing."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame(
            {
                "signal1": np.sin(np.linspace(0, 10, n)) + 0.1 * np.random.randn(n),
                "signal2": np.cos(np.linspace(0, 10, n)) + 0.1 * np.random.randn(n),
                "signal3": np.random.randn(n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1S"),
        )

    def test_apply_filter_workflow(self, sample_data: pd.DataFrame) -> None:
        """Test applying filter to signals."""
        processor = SignalProcessor()

        # Create filter config
        config = FilterConfig(
            filter_type="Moving Average",
            ma_window=10,
        )

        # Apply filter
        filtered_df = processor.apply_filter(sample_data, config)

        assert len(filtered_df) == len(sample_data)
        assert filtered_df.columns.tolist() == sample_data.columns.tolist()

        # Filtered signal should have lower variance
        for col in sample_data.columns:
            assert filtered_df[col].var() <= sample_data[col].var()

    def test_integration_workflow(self, sample_data: pd.DataFrame) -> None:
        """Test signal integration workflow."""
        processor = SignalProcessor()

        # Create integration config
        config = IntegrationConfig(
            signals_to_integrate=["signal1", "signal2"],
            integration_method="cumulative",
            initial_value=0.0,
        )

        # Apply integration
        result = processor.integrate_signals(sample_data, config)

        assert "signal1_integrated" in result.columns
        assert "signal2_integrated" in result.columns
        assert len(result) == len(sample_data)

    def test_differentiation_workflow(self, sample_data: pd.DataFrame) -> None:
        """Test signal differentiation workflow."""
        processor = SignalProcessor()

        # Create differentiation config
        config = DifferentiationConfig(
            signals_to_differentiate=["signal1", "signal2"],
            differentiation_order=1,
            method="central",
        )

        # Apply differentiation
        result = processor.differentiate_signals(sample_data, config)

        assert "signal1_deriv" in result.columns
        assert "signal2_deriv" in result.columns
        assert len(result) == len(sample_data)

    def test_custom_formula_workflow(self, sample_data: pd.DataFrame) -> None:
        """Test custom formula application."""
        processor = SignalProcessor()

        # Apply custom formula
        result_df, success = processor.apply_custom_formula(
            sample_data,
            "combined_signal",
            "signal1 + signal2",
        )

        assert success
        assert "combined_signal" in result_df.columns
        # Verify formula calculation
        expected = sample_data["signal1"] + sample_data["signal2"]
        pd.testing.assert_series_equal(result_df["combined_signal"], expected)

    def test_signal_statistics(self, sample_data: pd.DataFrame) -> None:
        """Test signal statistics calculation."""
        processor = SignalProcessor()

        stats = processor.detect_signal_statistics(sample_data, "signal1")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["count"] == 1000


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows."""

    @pytest.fixture()
    def workflow_data(self, tmp_path: Path) -> Path:
        """Create test data for end-to-end workflows."""
        np.random.seed(42)
        n = 1000

        # Create noisy sine wave data
        t = np.linspace(0, 10, n)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1S"),
                "temperature": 20
                + 5 * np.sin(2 * np.pi * t)
                + np.random.randn(n) * 0.5,
                "pressure": 100 + 10 * np.cos(2 * np.pi * t) + np.random.randn(n) * 1.0,
            },
        )

        csv_file = tmp_path / "workflow_data.csv"
        df.to_csv(csv_file, index=False)

        return csv_file

    def test_complete_processing_workflow(
        self, workflow_data: Path, tmp_path: Path,
    ) -> None:
        """Test complete data processing workflow."""
        # Step 1: Load data
        loader = DataLoader()
        df = loader.load_csv_file(str(workflow_data), validate_security=False)
        assert df is not None

        # Step 2: Convert time column
        time_col = loader.detect_time_column(df)
        df = loader.convert_time_column(df, time_col)

        # Step 3: Apply filtering
        processor = SignalProcessor()
        filter_config = FilterConfig(
            filter_type="Moving Average",
            ma_window=10,
        )
        filtered_df = processor.apply_filter(df, filter_config)

        # Step 4: Calculate statistics
        temp_stats = processor.detect_signal_statistics(filtered_df, "temperature")
        assert "mean" in temp_stats
        assert abs(temp_stats["mean"] - 20) < 1.0  # Should be close to 20

        # Step 5: Save results
        output_file = tmp_path / "processed_data.csv"
        success = loader.save_dataframe(filtered_df, str(output_file))
        assert success
        assert output_file.exists()

        # Verify saved data
        loaded = pd.read_csv(output_file)
        assert len(loaded) == len(filtered_df)

    def test_filter_integrate_differentiate_workflow(self, workflow_data: Path) -> None:
        """Test workflow with filter, integration, and differentiation."""
        # Load data
        loader = DataLoader()
        df = loader.load_csv_file(str(workflow_data), validate_security=False)

        # Convert time
        time_col = loader.detect_time_column(df)
        df = loader.convert_time_column(df, time_col)

        # Initialize processor
        processor = SignalProcessor()

        # Apply filter
        filter_config = FilterConfig(filter_type="Gaussian Filter", gaussian_sigma=2.0)
        df = processor.apply_filter(df, filter_config, signals=["temperature"])

        # Apply integration
        int_config = IntegrationConfig(
            signals_to_integrate=["pressure"],
            integration_method="cumulative",
        )
        df = processor.integrate_signals(df, int_config)

        # Apply differentiation
        diff_config = DifferentiationConfig(
            signals_to_differentiate=["temperature"],
            differentiation_order=1,
            method="central",
        )
        df = processor.differentiate_signals(df, diff_config)

        # Verify all operations completed
        assert "temperature" in df.columns
        assert "pressure_integrated" in df.columns
        assert "temperature_deriv" in df.columns

    def test_multiple_files_workflow(self, tmp_path: Path) -> None:
        """Test workflow with multiple input files."""
        # Create multiple test files
        np.random.seed(42)
        files = []

        for i in range(3):
            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        f"2024-01-0{i+1}", periods=100, freq="1min",
                    ),
                    f"sensor_{i+1}": np.random.randn(100) + i,
                    "common_sensor": np.random.randn(100),
                },
            )

            csv_file = tmp_path / f"multi_data_{i+1}.csv"
            df.to_csv(csv_file, index=False)
            files.append(str(csv_file))

        # Load all files
        loader = DataLoader()
        dataframes = loader.load_multiple_files(files)

        assert len(dataframes) == 3

        # Detect all signals
        all_signals = loader.detect_signals(files)

        assert "common_sensor" in all_signals
        assert len(all_signals) >= 4  # timestamp + 3 sensors + common

        # Process each DataFrame
        processor = SignalProcessor()
        filter_config = FilterConfig(filter_type="Median Filter", median_kernel=5)

        processed = {}
        for file_path, df in dataframes.items():
            filtered = processor.apply_filter(df, filter_config)
            processed[file_path] = filtered

        assert len(processed) == 3


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        loader = DataLoader()

        df = loader.load_csv_file("/nonexistent/file.csv", validate_security=False)

        assert df is None

    def test_invalid_filter_config(self) -> None:
        """Test handling invalid filter configuration."""
        processor = SignalProcessor()

        df = pd.DataFrame({"signal": np.random.randn(100)})

        # Create config with invalid filter type
        config = FilterConfig(filter_type="NonexistentFilter")

        # Should not crash, should return original data
        result = processor.apply_filter(df, config)

        pd.testing.assert_frame_equal(result, df)

    def test_custom_formula_error(self) -> None:
        """Test handling errors in custom formulas."""
        processor = SignalProcessor()

        df = pd.DataFrame({"signal1": np.random.randn(100)})

        # Invalid formula (references non-existent signal)
        result_df, success = processor.apply_custom_formula(
            df,
            "bad_signal",
            "nonexistent_signal + 1",
        )

        assert not success
        assert "bad_signal" not in result_df.columns


# Performance tests
class TestPerformance:
    """Performance integration tests."""

    @pytest.mark.slow()
    def test_large_dataset_workflow(self, tmp_path: Path) -> None:
        """Test workflow with large dataset."""
        import time

        # Create large dataset
        np.random.seed(42)
        n = 100_000

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1S"),
                "signal1": np.random.randn(n),
                "signal2": np.random.randn(n),
                "signal3": np.random.randn(n),
            },
        )

        csv_file = tmp_path / "large_data.csv"
        df.to_csv(csv_file, index=False)

        # Time the workflow
        start = time.perf_counter()

        # Load
        loader = DataLoader()
        df = loader.load_csv_file(str(csv_file), validate_security=False)

        # Process
        processor = SignalProcessor()
        filter_config = FilterConfig(filter_type="Moving Average", ma_window=10)
        filtered = processor.apply_filter(df, filter_config)

        # Save
        output = tmp_path / "large_output.csv"
        loader.save_dataframe(filtered, str(output))

        elapsed = time.perf_counter() - start

        # Should complete large dataset workflow in reasonable time
        assert elapsed < 10.0, f"Large dataset workflow too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
