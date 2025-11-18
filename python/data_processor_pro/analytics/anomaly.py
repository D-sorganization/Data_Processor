"""
Advanced anomaly detection algorithms for Data Processor Pro.

Provides multiple state-of-the-art anomaly detection methods including
statistical, machine learning, and time series approaches.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Professional anomaly detection tools.

    Features:
        - Statistical methods (Z-score, IQR, Modified Z-score)
        - Machine learning methods (Isolation Forest, LOF, DBSCAN)
        - Time series methods (STL decomposition, seasonal)
        - Ensemble methods (combining multiple detectors)
    """

    # ========== Statistical Methods ==========

    @staticmethod
    def z_score_method(data: np.ndarray,
                      threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.

        Args:
            data: Input data
            threshold: Z-score threshold

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > threshold

        logger.info(f"Z-score: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, z_scores

    @staticmethod
    def modified_z_score_method(data: np.ndarray,
                               threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using modified Z-score (MAD-based).

        More robust to outliers than standard Z-score.

        Args:
            data: Input data
            threshold: Modified Z-score threshold

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))

        modified_z_scores = 0.6745 * (data - median) / mad
        anomaly_mask = np.abs(modified_z_scores) > threshold

        logger.info(f"Modified Z-score: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, np.abs(modified_z_scores)

    @staticmethod
    def iqr_method(data: np.ndarray,
                  multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using IQR method.

        Args:
            data: Input data
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme)

        Returns:
            Tuple of (anomaly_mask, distance_from_bounds)
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        anomaly_mask = (data < lower_bound) | (data > upper_bound)

        # Distance from bounds
        distance = np.zeros_like(data)
        distance[data < lower_bound] = lower_bound - data[data < lower_bound]
        distance[data > upper_bound] = data[data > upper_bound] - upper_bound

        logger.info(f"IQR: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, distance

    @staticmethod
    def grubbs_test(data: np.ndarray,
                   alpha: float = 0.05) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Grubbs' test for detecting single outliers.

        Args:
            data: Input data
            alpha: Significance level

        Returns:
            Tuple of (anomaly_mask, test_results)
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Calculate Grubbs' test statistic
        max_deviation = np.max(np.abs(data - mean))
        g_stat = max_deviation / std

        # Critical value
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))

        # Find outlier
        anomaly_mask = np.abs(data - mean) >= max_deviation * 0.9999
        is_outlier = g_stat > g_crit

        results = {
            'g_statistic': g_stat,
            'g_critical': g_crit,
            'is_outlier': is_outlier,
            'outlier_value': data[anomaly_mask][0] if is_outlier else None,
        }

        logger.info(f"Grubbs test: {'Outlier' if is_outlier else 'No outlier'} detected")

        return anomaly_mask if is_outlier else np.zeros(n, dtype=bool), results

    # ========== Machine Learning Methods ==========

    @staticmethod
    def isolation_forest(data: np.ndarray,
                        contamination: float = 0.1,
                        n_estimators: int = 100,
                        random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isolation Forest anomaly detection.

        Args:
            data: Input data (1D or 2D array)
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise

        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )

        predictions = model.fit_predict(data)
        scores = -model.score_samples(data)  # Negative: higher = more anomalous

        anomaly_mask = predictions == -1

        logger.info(f"Isolation Forest: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, scores

    @staticmethod
    def local_outlier_factor(data: np.ndarray,
                            n_neighbors: int = 20,
                            contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local Outlier Factor (LOF) anomaly detection.

        Args:
            data: Input data (1D or 2D array)
            n_neighbors: Number of neighbors
            contamination: Expected proportion of anomalies

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise

        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )

        predictions = model.fit_predict(data)
        scores = -model.negative_outlier_factor_  # Negative: higher = more anomalous

        anomaly_mask = predictions == -1

        logger.info(f"LOF: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, scores

    @staticmethod
    def dbscan_anomaly(data: np.ndarray,
                      eps: float = 0.5,
                      min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        DBSCAN-based anomaly detection.

        Points not belonging to any cluster are considered anomalies.

        Args:
            data: Input data (1D or 2D array)
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood

        Returns:
            Tuple of (anomaly_mask, cluster_labels)
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise

        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)

        # Label -1 indicates noise/anomalies
        anomaly_mask = labels == -1

        logger.info(f"DBSCAN: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, labels

    # ========== Time Series Methods ==========

    @staticmethod
    def seasonal_decompose_anomaly(data: np.ndarray,
                                  period: int,
                                  threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect anomalies using seasonal decomposition.

        Args:
            data: Time series data
            period: Seasonal period
            threshold: Threshold for residual anomalies (in std deviations)

        Returns:
            Tuple of (anomaly_mask, decomposition_components)
        """
        # Simple seasonal decomposition
        if len(data) < 2 * period:
            raise ValueError("Data length must be at least 2 * period")

        # Compute trend (moving average)
        if period % 2 == 0:
            window = period
        else:
            window = period

        # Pad for edge handling
        trend = np.convolve(data, np.ones(window) / window, mode='same')

        # Detrend
        detrended = data - trend

        # Compute seasonal component
        seasonal = np.zeros_like(data)
        for i in range(period):
            indices = np.arange(i, len(data), period)
            if len(indices) > 0:
                seasonal[indices] = np.median(detrended[indices])

        # Residual
        residual = detrended - seasonal

        # Detect anomalies in residuals
        residual_std = np.std(residual)
        anomaly_mask = np.abs(residual) > threshold * residual_std

        components = {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
        }

        logger.info(f"Seasonal decomposition: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, components

    @staticmethod
    def rolling_statistics_anomaly(data: np.ndarray,
                                   window: int = 20,
                                   n_sigma: float = 3.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect anomalies using rolling statistics.

        Args:
            data: Time series data
            window: Rolling window size
            n_sigma: Number of standard deviations for threshold

        Returns:
            Tuple of (anomaly_mask, rolling_statistics)
        """
        from numpy.lib.stride_tricks import sliding_window_view

        if window > len(data):
            raise ValueError("Window size cannot exceed data length")

        # Compute rolling statistics
        windows = sliding_window_view(data, window)
        rolling_mean = np.mean(windows, axis=1)
        rolling_std = np.std(windows, axis=1, ddof=1)

        # Pad to match original length
        pad_size = window - 1
        rolling_mean = np.concatenate([np.full(pad_size, rolling_mean[0]), rolling_mean])
        rolling_std = np.concatenate([np.full(pad_size, rolling_std[0]), rolling_std])

        # Detect anomalies
        lower_bound = rolling_mean - n_sigma * rolling_std
        upper_bound = rolling_mean + n_sigma * rolling_std

        anomaly_mask = (data < lower_bound) | (data > upper_bound)

        stats_dict = {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        }

        logger.info(f"Rolling statistics: {np.sum(anomaly_mask)} anomalies detected")

        return anomaly_mask, stats_dict

    # ========== Ensemble Methods ==========

    @staticmethod
    def ensemble_anomaly(data: np.ndarray,
                        methods: Optional[list] = None,
                        voting: str = 'majority') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Ensemble anomaly detection combining multiple methods.

        Args:
            data: Input data
            methods: List of methods to use (default: all statistical)
            voting: 'majority' or 'unanimous'

        Returns:
            Tuple of (anomaly_mask, individual_results)
        """
        if methods is None:
            methods = ['z_score', 'modified_z_score', 'iqr']

        detector = AnomalyDetector()
        results = {}

        # Apply each method
        for method in methods:
            if method == 'z_score':
                mask, _ = detector.z_score_method(data)
            elif method == 'modified_z_score':
                mask, _ = detector.modified_z_score_method(data)
            elif method == 'iqr':
                mask, _ = detector.iqr_method(data)
            else:
                logger.warning(f"Unknown method: {method}")
                continue

            results[method] = mask

        # Combine results
        if voting == 'majority':
            # At least half of methods agree
            votes = np.sum(list(results.values()), axis=0)
            threshold = len(results) / 2
            anomaly_mask = votes > threshold
        elif voting == 'unanimous':
            # All methods agree
            anomaly_mask = np.all(list(results.values()), axis=0)
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

        logger.info(
            f"Ensemble ({voting}): {np.sum(anomaly_mask)} anomalies detected "
            f"from {len(methods)} methods"
        )

        return anomaly_mask, results

    # ========== Utility Methods ==========

    @staticmethod
    def get_anomaly_indices(anomaly_mask: np.ndarray) -> np.ndarray:
        """Get indices of anomalies."""
        return np.where(anomaly_mask)[0]

    @staticmethod
    def get_anomaly_values(data: np.ndarray, anomaly_mask: np.ndarray) -> np.ndarray:
        """Get values of anomalies."""
        return data[anomaly_mask]

    @staticmethod
    def remove_anomalies(data: np.ndarray, anomaly_mask: np.ndarray) -> np.ndarray:
        """Remove anomalies from data."""
        return data[~anomaly_mask]

    @staticmethod
    def replace_anomalies(data: np.ndarray, anomaly_mask: np.ndarray,
                         method: str = 'median') -> np.ndarray:
        """
        Replace anomalies with substitute values.

        Args:
            data: Input data
            anomaly_mask: Boolean mask of anomalies
            method: Replacement method ('median', 'mean', 'interpolate')

        Returns:
            Data with anomalies replaced
        """
        result = data.copy()

        if method == 'median':
            replacement = np.median(data[~anomaly_mask])
            result[anomaly_mask] = replacement
        elif method == 'mean':
            replacement = np.mean(data[~anomaly_mask])
            result[anomaly_mask] = replacement
        elif method == 'interpolate':
            indices = np.arange(len(data))
            valid_mask = ~anomaly_mask
            result[anomaly_mask] = np.interp(
                indices[anomaly_mask],
                indices[valid_mask],
                data[valid_mask]
            )
        else:
            raise ValueError(f"Unknown replacement method: {method}")

        return result
