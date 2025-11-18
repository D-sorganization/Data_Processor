"""
Statistical analysis tools for professional data analysis.

Provides comprehensive statistical tests, correlation analysis,
distribution fitting, and hypothesis testing.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Professional statistical analysis tools.

    Features:
        - Descriptive statistics
        - Hypothesis testing (t-test, ANOVA, chi-square)
        - Correlation analysis (Pearson, Spearman, Kendall)
        - Distribution fitting
        - Time series decomposition
        - Normality tests
        - Outlier detection
    """

    @staticmethod
    def descriptive_stats(data: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive descriptive statistics.

        Args:
            data: Input data array

        Returns:
            Dictionary of statistics
        """
        return {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'var': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.ptp(data),
            'q25': np.percentile(data, 25),
            'q50': np.percentile(data, 50),
            'q75': np.percentile(data, 75),
            'iqr': stats.iqr(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.inf,
        }

    @staticmethod
    def correlation_matrix(data: np.ndarray, method: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute correlation matrix.

        Args:
            data: 2D array where each column is a variable
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Tuple of (correlation_matrix, p_values)
        """
        n_vars = data.shape[1]
        corr_matrix = np.zeros((n_vars, n_vars))
        p_values = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    if method == 'pearson':
                        corr, pval = stats.pearsonr(data[:, i], data[:, j])
                    elif method == 'spearman':
                        corr, pval = stats.spearmanr(data[:, i], data[:, j])
                    elif method == 'kendall':
                        corr, pval = stats.kendalltau(data[:, i], data[:, j])
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")

                    corr_matrix[i, j] = corr
                    p_values[i, j] = pval

        return corr_matrix, p_values

    @staticmethod
    def t_test(sample1: np.ndarray, sample2: np.ndarray,
              equal_var: bool = True) -> Dict[str, Any]:
        """
        Perform independent t-test.

        Args:
            sample1: First sample
            sample2: Second sample
            equal_var: Assume equal variance (Student's t-test if True, Welch's if False)

        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean1': np.mean(sample1),
            'mean2': np.mean(sample2),
            'std1': np.std(sample1, ddof=1),
            'std2': np.std(sample2, ddof=1),
            'n1': len(sample1),
            'n2': len(sample2),
        }

    @staticmethod
    def anova(*groups) -> Dict[str, Any]:
        """
        Perform one-way ANOVA.

        Args:
            *groups: Variable number of sample groups

        Returns:
            Dictionary with test results
        """
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_groups': len(groups),
            'group_means': [np.mean(g) for g in groups],
            'group_stds': [np.std(g, ddof=1) for g in groups],
        }

    @staticmethod
    def chi_square_test(observed: np.ndarray,
                       expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform chi-square goodness-of-fit test.

        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)

        Returns:
            Dictionary with test results
        """
        if expected is None:
            expected = np.ones_like(observed) * np.mean(observed)

        chi2, p_value = stats.chisquare(observed, expected)

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'degrees_of_freedom': len(observed) - 1,
        }

    @staticmethod
    def normality_test(data: np.ndarray) -> Dict[str, Any]:
        """
        Test for normality using multiple tests.

        Args:
            data: Input data

        Returns:
            Dictionary with test results
        """
        # Shapiro-Wilk test (best for small samples)
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))

        # Anderson-Darling test
        anderson_result = stats.anderson(data, dist='norm')

        # D'Agostino-Pearson test
        try:
            k2_stat, k2_p = stats.normaltest(data)
        except:
            k2_stat, k2_p = np.nan, np.nan

        return {
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal': shapiro_p > 0.05,
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'normal': ks_p > 0.05,
            },
            'anderson_darling': {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_levels': anderson_result.significance_level.tolist(),
            },
            'dagostino_pearson': {
                'statistic': k2_stat,
                'p_value': k2_p,
                'normal': k2_p > 0.05 if not np.isnan(k2_p) else None,
            },
        }

    @staticmethod
    def fit_distribution(data: np.ndarray,
                        distributions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit multiple distributions and find best fit.

        Args:
            data: Input data
            distributions: List of distribution names to try

        Returns:
            Dictionary with fitting results
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min']

        results = {}
        best_dist = None
        best_aic = np.inf

        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)

                # Fit distribution
                params = dist.fit(data)

                # Calculate AIC (Akaike Information Criterion)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, dist_name, args=params)

                results[dist_name] = {
                    'parameters': params,
                    'aic': aic,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                }

                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist_name

            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")

        return {
            'distributions': results,
            'best_fit': best_dist,
            'best_aic': best_aic,
        }

    @staticmethod
    def outlier_detection_iqr(data: np.ndarray,
                             multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using IQR method.

        Args:
            data: Input data
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)

        Returns:
            Tuple of (outlier_mask, outlier_indices)
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]

        return outlier_mask, outlier_indices

    @staticmethod
    def moving_statistics(data: np.ndarray, window: int = 10) -> Dict[str, np.ndarray]:
        """
        Compute moving statistics.

        Args:
            data: Input data
            window: Window size

        Returns:
            Dictionary of moving statistics
        """
        from numpy.lib.stride_tricks import sliding_window_view

        if window > len(data):
            raise ValueError("Window size cannot be larger than data length")

        windows = sliding_window_view(data, window)

        return {
            'mean': np.mean(windows, axis=1),
            'std': np.std(windows, axis=1, ddof=1),
            'min': np.min(windows, axis=1),
            'max': np.max(windows, axis=1),
            'median': np.median(windows, axis=1),
        }

    @staticmethod
    def autocorrelation(data: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation function.

        Args:
            data: Input data
            max_lag: Maximum lag (default: len(data) // 2)

        Returns:
            Autocorrelation values
        """
        if max_lag is None:
            max_lag = len(data) // 2

        # Normalize data
        data_normalized = (data - np.mean(data)) / np.std(data)

        # Compute autocorrelation
        acf = np.correlate(data_normalized, data_normalized, mode='full')
        acf = acf[len(acf) // 2:]
        acf = acf / acf[0]  # Normalize to 1 at lag 0

        return acf[:max_lag + 1]

    @staticmethod
    def confidence_interval(data: np.ndarray,
                          confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute confidence interval for mean.

        Args:
            data: Input data
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

        return mean, mean - margin, mean + margin

    @staticmethod
    def time_series_decomposition(data: np.ndarray,
                                  period: int) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            data: Time series data
            period: Seasonal period

        Returns:
            Dictionary with components
        """
        from scipy.signal import detrend

        # Compute trend (moving average)
        if period % 2 == 0:
            window = period
        else:
            window = period

        trend = np.convolve(data, np.ones(window) / window, mode='same')

        # Detrend to get seasonal + residual
        detrended = data - trend

        # Compute seasonal component (average over periods)
        seasonal = np.zeros_like(data)
        for i in range(period):
            indices = np.arange(i, len(data), period)
            seasonal[indices] = np.mean(detrended[indices])

        # Residual
        residual = detrended - seasonal

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'original': data,
        }
