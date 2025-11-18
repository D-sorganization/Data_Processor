"""
Basic tests for Data Processor Pro.

Tests core functionality to ensure everything works correctly.
"""

import pytest
import numpy as np
from pathlib import Path

from data_processor_pro.core import DataEngine, SignalProcessor
from data_processor_pro.analytics import StatisticalAnalyzer, AnomalyDetector, MLPreprocessor
from data_processor_pro.config import AppConfig, ConfigLoader


class TestDataEngine:
    """Test DataEngine functionality."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = DataEngine(use_polars=True, max_workers=2)
        assert engine.use_polars == True
        assert engine.max_workers == 2

    def test_data_operations(self):
        """Test basic data operations."""
        engine = DataEngine()

        # Test shape and columns
        assert engine.shape() == (0, 0)
        assert engine.columns() == []


class TestSignalProcessor:
    """Test SignalProcessor functionality."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return SignalProcessor(use_numba=True)

    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        np.random.seed(42)
        return np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)

    def test_moving_average(self, processor, sample_signal):
        """Test moving average filter."""
        filtered = processor.moving_average(sample_signal, window_size=5)
        assert len(filtered) == len(sample_signal)
        assert np.std(filtered) < np.std(sample_signal)  # Should reduce noise

    def test_butterworth_lowpass(self, processor, sample_signal):
        """Test Butterworth low-pass filter."""
        filtered = processor.butterworth_lowpass(sample_signal, cutoff=0.1, order=4)
        assert len(filtered) == len(sample_signal)

    def test_median_filter(self, processor, sample_signal):
        """Test median filter."""
        filtered = processor.median_filter(sample_signal, kernel_size=5)
        assert len(filtered) == len(sample_signal)

    def test_gaussian_filter(self, processor, sample_signal):
        """Test Gaussian filter."""
        filtered = processor.gaussian_filter(sample_signal, sigma=1.0)
        assert len(filtered) == len(sample_signal)

    def test_kalman_filter(self, processor, sample_signal):
        """Test Kalman filter (new feature)."""
        filtered = processor.kalman_filter(sample_signal)
        assert len(filtered) == len(sample_signal)
        # Kalman should smooth the signal
        assert np.std(filtered) < np.std(sample_signal)

    def test_integration(self, processor):
        """Test integration."""
        signal = np.ones(100)
        integrated = processor.integrate(signal, dt=1.0, method='cumulative')
        assert len(integrated) == len(signal)
        # Integration of constant should be linear
        assert np.allclose(integrated, np.arange(len(signal)), atol=1.0)

    def test_differentiation(self, processor):
        """Test differentiation."""
        signal = np.arange(100, dtype=float)
        diff = processor.differentiate(signal, dt=1.0, method='central')
        # Derivative of linear function should be constant
        assert np.allclose(diff[1:-1], 1.0, atol=0.1)

    def test_normalize(self, processor):
        """Test normalization."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Min-max normalization
        norm = processor.normalize(signal, method='minmax')
        assert norm.min() == 0.0
        assert norm.max() == 1.0

        # Z-score normalization
        norm = processor.normalize(signal, method='zscore')
        assert abs(norm.mean()) < 1e-10
        assert abs(norm.std() - 1.0) < 1e-10


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StatisticalAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.normal(100, 15, 1000)

    def test_descriptive_stats(self, analyzer, sample_data):
        """Test descriptive statistics."""
        stats = analyzer.descriptive_stats(sample_data)

        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

        # Check reasonable values
        assert 95 < stats['mean'] < 105
        assert 95 < stats['median'] < 105

    def test_normality_test(self, analyzer, sample_data):
        """Test normality testing."""
        results = analyzer.normality_test(sample_data)

        assert 'shapiro_wilk' in results
        assert 'kolmogorov_smirnov' in results
        # Normal data should pass normality tests
        assert results['shapiro_wilk']['normal'] == True

    def test_correlation_matrix(self, analyzer):
        """Test correlation matrix."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        corr_matrix, p_values = analyzer.correlation_matrix(data)

        assert corr_matrix.shape == (3, 3)
        assert p_values.shape == (3, 3)
        # Diagonal should be 1.0
        assert np.allclose(np.diag(corr_matrix), 1.0)

    def test_t_test(self, analyzer):
        """Test t-test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 100)
        sample2 = np.random.normal(105, 10, 100)

        results = analyzer.t_test(sample1, sample2)

        assert 't_statistic' in results
        assert 'p_value' in results
        assert 'significant' in results


class TestAnomalyDetector:
    """Test AnomalyDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return AnomalyDetector()

    @pytest.fixture
    def sample_data_with_outliers(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 1000)
        # Add some outliers
        data[50] = 200
        data[100] = 0
        data[500] = 250
        return data

    def test_z_score_method(self, detector, sample_data_with_outliers):
        """Test Z-score anomaly detection."""
        anomalies, scores = detector.z_score_method(sample_data_with_outliers, threshold=3.0)

        assert len(anomalies) == len(sample_data_with_outliers)
        assert len(scores) == len(sample_data_with_outliers)
        # Should detect the outliers
        assert anomalies.sum() >= 3

    def test_iqr_method(self, detector, sample_data_with_outliers):
        """Test IQR anomaly detection."""
        anomalies, distances = detector.iqr_method(sample_data_with_outliers, multiplier=1.5)

        assert len(anomalies) == len(sample_data_with_outliers)
        assert anomalies.sum() >= 3

    def test_replace_anomalies(self, detector, sample_data_with_outliers):
        """Test anomaly replacement."""
        anomalies, _ = detector.z_score_method(sample_data_with_outliers, threshold=3.0)

        # Replace with median
        replaced = detector.replace_anomalies(
            sample_data_with_outliers,
            anomalies,
            method='median'
        )

        # Replaced values should be close to median
        median = np.median(sample_data_with_outliers[~anomalies])
        assert np.allclose(replaced[anomalies], median)


class TestMLPreprocessor:
    """Test MLPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return MLPreprocessor()

    @pytest.fixture
    def sample_data_2d(self):
        """Create 2D sample data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_standard_scaler(self, preprocessor, sample_data_2d):
        """Test standard scaling."""
        scaled = preprocessor.standard_scaler(sample_data_2d, fit=True)

        # Mean should be ~0, std should be ~1
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-10)

    def test_minmax_scaler(self, preprocessor, sample_data_2d):
        """Test min-max scaling."""
        scaled = preprocessor.minmax_scaler(sample_data_2d, fit=True)

        # Should be in range [0, 1]
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_pca(self, preprocessor, sample_data_2d):
        """Test PCA dimensionality reduction."""
        transformed, components, explained_var = preprocessor.pca(
            sample_data_2d,
            n_components=3
        )

        assert transformed.shape == (100, 3)
        assert components.shape == (5, 3)
        assert len(explained_var) == 3
        # Variance should sum to <= 1.0
        assert explained_var.sum() <= 1.0

    def test_polynomial_features(self, preprocessor):
        """Test polynomial feature generation."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        poly = preprocessor.polynomial_features(data, degree=2)

        # Original features + squared features
        assert poly.shape[1] == 4  # 2 original + 2 squared


class TestConfiguration:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = ConfigLoader.get_default_config()

        assert config.version == "1.0.0"
        assert config.performance.use_polars == True
        assert config.performance.use_numba == True
        assert config.ui.theme.value == "dark"

    def test_config_validation(self):
        """Test configuration validation."""
        config = AppConfig()
        config.validate()  # Should not raise

        # Invalid config
        config.performance.max_workers = -1
        with pytest.raises(ValueError):
            config.validate()


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    def test_filter_performance(self, benchmark):
        """Benchmark filter performance."""
        processor = SignalProcessor(use_numba=True)
        signal = np.random.randn(100000)

        result = benchmark(processor.moving_average, signal, window_size=10)
        assert len(result) == len(signal)

    def test_kalman_performance(self, benchmark):
        """Benchmark Kalman filter performance."""
        processor = SignalProcessor(use_numba=True)
        signal = np.random.randn(100000)

        result = benchmark(processor.kalman_filter, signal)
        assert len(result) == len(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
