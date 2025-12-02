"""Comprehensive test suite for vectorized filter engine.

This test suite follows The Pragmatic Programmer principles:
- Design by Contract: Tests verify preconditions and postconditions
- Crash Early: Tests ensure filters fail gracefully with invalid input
- Test Your Software: Comprehensive coverage of all filter types

Run with: pytest test_vectorized_filters.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path so we can import data_processor package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor.vectorized_filter_engine import VectorizedFilterEngine


class TestVectorizedFilterEngine:
    """Test suite for VectorizedFilterEngine."""

    @pytest.fixture
    def engine(self) -> VectorizedFilterEngine:
        """Create a filter engine instance."""
        return VectorizedFilterEngine()

    @pytest.fixture
    def simple_signal(self) -> pd.Series:
        """Generate simple test signal."""
        return pd.Series(np.arange(100, dtype=float))

    @pytest.fixture
    def noisy_sine_wave(self) -> pd.Series:
        """Generate noisy sine wave for filter testing."""
        np.random.seed(42)
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)
        return pd.Series(signal)

    @pytest.fixture
    def signal_with_outliers(self) -> pd.Series:
        """Generate signal with outliers for outlier detection tests."""
        np.random.seed(42)
        signal = np.random.randn(100)
        # Add outliers
        signal[10] = 10.0
        signal[50] = -10.0
        signal[90] = 15.0
        return pd.Series(signal)

    @pytest.fixture
    def multi_signal_df(self) -> pd.DataFrame:
        """Generate DataFrame with multiple signals."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "signal1": np.random.randn(1000),
                "signal2": np.random.randn(1000),
                "signal3": np.random.randn(1000),
            },
        )

    # =================================================================
    # Moving Average Filter Tests
    # =================================================================

    def test_moving_average_reduces_noise(
        self,
        engine: VectorizedFilterEngine,
        noisy_sine_wave: pd.Series,
    ) -> None:
        """Test that moving average reduces signal variance."""
        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(noisy_sine_wave, params)

        assert filtered.var() < noisy_sine_wave.var()
        assert len(filtered) == len(noisy_sine_wave)

    def test_moving_average_preserves_mean(
        self,
        engine: VectorizedFilterEngine,
        noisy_sine_wave: pd.Series,
    ) -> None:
        """Test that moving average preserves signal mean."""
        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(noisy_sine_wave, params)

        np.testing.assert_allclose(filtered.mean(), noisy_sine_wave.mean(), rtol=0.01)

    @pytest.mark.parametrize("window", [3, 5, 11, 21, 51])
    def test_moving_average_window_sizes(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
        window: int,
    ) -> None:
        """Test moving average with various window sizes."""
        params = {"ma_window": window}
        filtered = engine._apply_moving_average_vectorized(simple_signal, params)

        assert len(filtered) == len(simple_signal)
        assert not filtered.isnull().all()

    def test_moving_average_handles_short_signal(
        self,
        engine: VectorizedFilterEngine,
    ) -> None:
        """Test moving average with signal shorter than window."""
        short_signal = pd.Series([1.0, 2.0])
        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(short_signal, params)

        # Should return original signal
        pd.testing.assert_series_equal(filtered, short_signal)

    # =================================================================
    # Butterworth Filter Tests
    # =================================================================

    def test_butterworth_lowpass_attenuates_high_frequency(
        self,
        engine: VectorizedFilterEngine,
    ) -> None:
        """Test Butterworth low-pass filter removes high frequencies."""
        # Create signal with known frequency components
        t = pd.date_range("2000", periods=1000, freq="1ms")
        low_freq = np.sin(2 * np.pi * 5 * np.arange(1000) / 1000)  # 5 Hz
        high_freq = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000)  # 50 Hz
        signal = pd.Series(low_freq + high_freq, index=t)

        params = {
            "bw_order": 4,
            "bw_cutoff": 0.2,
            "filter_type": "Butterworth Low-pass",
        }
        filtered = engine._apply_butterworth_vectorized(signal, params)

        # High frequency should be attenuated
        fft_original = np.fft.fft(signal.values)
        fft_filtered = np.fft.fft(filtered.values)

        # Check that high frequency component is reduced
        assert np.abs(fft_filtered[50]) < 0.5 * np.abs(fft_original[50])

    def test_butterworth_handles_insufficient_data(
        self,
        engine: VectorizedFilterEngine,
    ) -> None:
        """Test Butterworth filter with insufficient data points."""
        short_signal = pd.Series([1.0, 2.0, 3.0])
        params = {
            "bw_order": 4,
            "bw_cutoff": 0.1,
            "filter_type": "Butterworth Low-pass",
        }
        filtered = engine._apply_butterworth_vectorized(short_signal, params)

        # Should return original signal
        pd.testing.assert_series_equal(filtered, short_signal)

    # =================================================================
    # Median Filter Tests
    # =================================================================

    def test_median_filter_removes_outliers(
        self,
        engine: VectorizedFilterEngine,
        signal_with_outliers: pd.Series,
    ) -> None:
        """Test median filter removes outliers."""
        params = {"median_kernel": 5}
        filtered = engine._apply_median_vectorized(signal_with_outliers, params)

        # Outliers should be reduced
        assert abs(filtered.iloc[10]) < abs(signal_with_outliers.iloc[10])
        assert abs(filtered.iloc[50]) < abs(signal_with_outliers.iloc[50])

    def test_median_filter_ensures_odd_kernel(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
    ) -> None:
        """Test median filter converts even kernel to odd."""
        params = {"median_kernel": 6}  # Even number
        filtered = engine._apply_median_vectorized(simple_signal, params)

        # Should still work (kernel converted to 7)
        assert len(filtered) == len(simple_signal)

    # =================================================================
    # Hampel Filter Tests
    # =================================================================

    def test_hampel_filter_identifies_outliers(
        self,
        engine: VectorizedFilterEngine,
        signal_with_outliers: pd.Series,
    ) -> None:
        """Test Hampel filter identifies and replaces outliers."""
        params = {"hampel_window": 5, "hampel_threshold": 3.0}
        filtered = engine._apply_hampel_vectorized(signal_with_outliers, params)

        # Outliers should be replaced
        assert abs(filtered.iloc[10] - signal_with_outliers.iloc[10]) > 5.0
        assert abs(filtered.iloc[50] - signal_with_outliers.iloc[50]) > 5.0

    # =================================================================
    # Z-Score Filter Tests
    # =================================================================

    @pytest.mark.parametrize("method", ["standard", "modified"])
    def test_zscore_filter_removes_outliers(
        self,
        engine: VectorizedFilterEngine,
        signal_with_outliers: pd.Series,
        method: str,
    ) -> None:
        """Test Z-score filter removes outliers."""
        params = {"zscore_threshold": 3.0, "zscore_method": method}
        filtered = engine._apply_zscore_vectorized(signal_with_outliers, params)

        # Outliers should be marked as NaN
        assert pd.isna(filtered.iloc[10])
        assert pd.isna(filtered.iloc[50])
        assert pd.isna(filtered.iloc[90])

    def test_zscore_handles_zero_std(self, engine: VectorizedFilterEngine) -> None:
        """Test Z-score filter handles constant signal."""
        constant_signal = pd.Series([5.0] * 100)
        params = {"zscore_threshold": 3.0, "zscore_method": "standard"}
        filtered = engine._apply_zscore_vectorized(constant_signal, params)

        # Should return original signal
        pd.testing.assert_series_equal(filtered, constant_signal)

    # =================================================================
    # Savitzky-Golay Filter Tests
    # =================================================================

    def test_savgol_filter_smooths_signal(
        self,
        engine: VectorizedFilterEngine,
        noisy_sine_wave: pd.Series,
    ) -> None:
        """Test Savitzky-Golay filter smooths signal."""
        params = {"savgol_window": 11, "savgol_polyorder": 2}
        filtered = engine._apply_savgol_vectorized(noisy_sine_wave, params)

        assert filtered.var() < noisy_sine_wave.var()
        assert len(filtered) == len(noisy_sine_wave)

    def test_savgol_ensures_odd_window(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
    ) -> None:
        """Test Savitzky-Golay converts even window to odd."""
        params = {"savgol_window": 10, "savgol_polyorder": 2}  # Even number
        filtered = engine._apply_savgol_vectorized(simple_signal, params)

        # Should still work (window converted to 11)
        assert len(filtered) == len(simple_signal)

    def test_savgol_handles_invalid_polyorder(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
    ) -> None:
        """Test Savitzky-Golay handles polyorder >= window."""
        params = {"savgol_window": 11, "savgol_polyorder": 12}  # Too high
        filtered = engine._apply_savgol_vectorized(simple_signal, params)

        # Should adjust polyorder automatically
        assert len(filtered) == len(simple_signal)

    # =================================================================
    # Gaussian Filter Tests
    # =================================================================

    def test_gaussian_filter_smooths_signal(
        self,
        engine: VectorizedFilterEngine,
        noisy_sine_wave: pd.Series,
    ) -> None:
        """Test Gaussian filter smooths signal."""
        params = {"gaussian_sigma": 2.0, "gaussian_mode": "reflect"}
        filtered = engine._apply_gaussian_vectorized(noisy_sine_wave, params)

        assert filtered.var() < noisy_sine_wave.var()
        assert len(filtered) == len(noisy_sine_wave)

    # =================================================================
    # Batch Processing Tests
    # =================================================================

    def test_batch_processing_consistency(
        self,
        engine: VectorizedFilterEngine,
        multi_signal_df: pd.DataFrame,
    ) -> None:
        """Test that batch processing gives same results as individual."""
        params = {"ma_window": 5}

        # Batch process
        batch_result = engine.apply_filter_batch(
            multi_signal_df,
            "Moving Average",
            params,
        )

        # Individual process
        individual_results = pd.DataFrame()
        for col in multi_signal_df.columns:
            individual_results[col] = engine._apply_single_filter(
                multi_signal_df[col],
                "Moving Average",
                params,
                col,
            )

        pd.testing.assert_frame_equal(batch_result, individual_results)

    def test_parallel_vs_sequential(
        self,
        engine: VectorizedFilterEngine,
        multi_signal_df: pd.DataFrame,
    ) -> None:
        """Test parallel processing gives same results as sequential."""
        params = {"ma_window": 10}

        # Parallel
        engine.n_jobs = 4
        parallel_result = engine.apply_filter_batch(
            multi_signal_df,
            "Moving Average",
            params,
        )

        # Sequential
        engine.n_jobs = 1
        sequential_result = engine.apply_filter_batch(
            multi_signal_df,
            "Moving Average",
            params,
        )

        pd.testing.assert_frame_equal(parallel_result, sequential_result)

    # =================================================================
    # Edge Case Tests
    # =================================================================

    def test_filter_handles_nan_values(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
    ) -> None:
        """Test that filters handle NaN values gracefully."""
        signal_with_nan = simple_signal.copy()
        signal_with_nan.iloc[10:20] = np.nan

        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(signal_with_nan, params)

        # Should not crash and should return same length
        assert len(filtered) == len(signal_with_nan)

    def test_filter_handles_empty_signal(self, engine: VectorizedFilterEngine) -> None:
        """Test filter handles empty signal."""
        empty_signal = pd.Series([])
        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(empty_signal, params)

        # Should return empty signal
        assert len(filtered) == 0

    def test_filter_handles_single_value(self, engine: VectorizedFilterEngine) -> None:
        """Test filter handles single-value signal."""
        single_signal = pd.Series([5.0])
        params = {"ma_window": 10}
        filtered = engine._apply_moving_average_vectorized(single_signal, params)

        # Should return original signal
        pd.testing.assert_series_equal(filtered, single_signal)

    def test_invalid_filter_type(
        self,
        engine: VectorizedFilterEngine,
        simple_signal: pd.Series,
    ) -> None:
        """Test handling of invalid filter type."""
        result = engine.apply_filter_batch(
            pd.DataFrame({"sig": simple_signal}),
            "NonexistentFilter",
            {},
        )

        # Should return original DataFrame
        pd.testing.assert_frame_equal(result, pd.DataFrame({"sig": simple_signal}))

    # =================================================================
    # Parameter Validation Tests
    # =================================================================

    def test_safe_get_param_with_string(self, engine: VectorizedFilterEngine) -> None:
        """Test parameter extraction with string values."""
        params = {"ma_window": "10"}
        value = engine._safe_get_param(params, "ma_window", 5)

        assert value == 10.0

    def test_safe_get_param_with_invalid_string(
        self,
        engine: VectorizedFilterEngine,
    ) -> None:
        """Test parameter extraction with invalid string."""
        params = {"ma_window": "invalid"}
        value = engine._safe_get_param(params, "ma_window", 5)

        assert value == 5  # Should use default

    def test_safe_get_param_clamping(self, engine: VectorizedFilterEngine) -> None:
        """Test parameter value clamping."""
        # Too small
        assert engine._safe_get_param({}, "x", 5, min_val=10, max_val=20) == 10

        # Too large
        assert engine._safe_get_param({}, "x", 25, min_val=10, max_val=20) == 20

        # Just right
        assert engine._safe_get_param({}, "x", 15, min_val=10, max_val=20) == 15


# =================================================================
# Performance Tests
# =================================================================


class TestFilterPerformance:
    """Performance regression tests for filters."""

    @pytest.fixture
    def large_signal(self) -> pd.Series:
        """Generate large signal for performance testing."""
        np.random.seed(42)
        return pd.Series(np.random.randn(1_000_000))

    @pytest.mark.slow
    def test_moving_average_performance(self, large_signal: pd.Series) -> None:
        """Test moving average processes 1M points quickly."""
        import time

        engine = VectorizedFilterEngine()
        params = {"ma_window": 10}

        start = time.perf_counter()
        _ = engine._apply_moving_average_vectorized(large_signal, params)
        elapsed = time.perf_counter() - start

        # Should process 1M points in < 1 second
        assert elapsed < 1.0, f"Moving average too slow: {elapsed:.2f}s"

    @pytest.mark.slow
    def test_batch_processing_scales(self) -> None:
        """Test batch processing scales well with multiple signals."""
        import time

        engine = VectorizedFilterEngine()
        df = pd.DataFrame({f"signal{i}": np.random.randn(100_000) for i in range(10)})

        params = {"ma_window": 10}

        start = time.perf_counter()
        _ = engine.apply_filter_batch(df, "Moving Average", params)
        elapsed = time.perf_counter() - start

        # Should process 10 signals of 100k points in < 2 seconds
        assert elapsed < 2.0, f"Batch processing too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
