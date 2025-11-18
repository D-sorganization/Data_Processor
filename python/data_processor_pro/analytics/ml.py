"""
Machine Learning preprocessing tools for Data Processor Pro.

Provides professional ML preprocessing including scaling, encoding,
feature engineering, and dimensionality reduction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalerParams:
    """Parameters for fitted scaler."""
    method: str
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    mad: Optional[np.ndarray] = None


class MLPreprocessor:
    """
    Machine Learning preprocessing tools.

    Features:
        - Multiple scaling methods (StandardScaler, MinMaxScaler, RobustScaler)
        - Encoding (one-hot, label, ordinal)
        - Feature engineering (polynomial, interaction)
        - Dimensionality reduction (PCA, normalization)
        - Missing value imputation
        - Feature selection
    """

    def __init__(self):
        """Initialize ML preprocessor."""
        self._scaler_params: Optional[ScalerParams] = None

    # ========== Scaling Methods ==========

    def standard_scaler(self, data: np.ndarray,
                       fit: bool = True) -> np.ndarray:
        """
        Apply StandardScaler (z-score normalization).

        Args:
            data: Input data (2D array)
            fit: Whether to fit parameters

        Returns:
            Scaled data
        """
        if fit:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1)
            self._scaler_params = ScalerParams(method='standard', mean=mean, std=std)
        else:
            if self._scaler_params is None or self._scaler_params.method != 'standard':
                raise ValueError("Scaler not fitted. Set fit=True first.")
            mean = self._scaler_params.mean
            std = self._scaler_params.std

        # Avoid division by zero
        std = np.where(std == 0, 1, std)

        return (data - mean) / std

    def minmax_scaler(self, data: np.ndarray,
                     feature_range: Tuple[float, float] = (0, 1),
                     fit: bool = True) -> np.ndarray:
        """
        Apply MinMaxScaler.

        Args:
            data: Input data (2D array)
            feature_range: Target range (min, max)
            fit: Whether to fit parameters

        Returns:
            Scaled data
        """
        if fit:
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            self._scaler_params = ScalerParams(method='minmax', min=min_val, max=max_val)
        else:
            if self._scaler_params is None or self._scaler_params.method != 'minmax':
                raise ValueError("Scaler not fitted. Set fit=True first.")
            min_val = self._scaler_params.min
            max_val = self._scaler_params.max

        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)

        # Scale to [0, 1]
        data_scaled = (data - min_val) / range_val

        # Scale to target range
        target_min, target_max = feature_range
        return data_scaled * (target_max - target_min) + target_min

    def robust_scaler(self, data: np.ndarray,
                     fit: bool = True) -> np.ndarray:
        """
        Apply RobustScaler (median and MAD).

        More robust to outliers than StandardScaler.

        Args:
            data: Input data (2D array)
            fit: Whether to fit parameters

        Returns:
            Scaled data
        """
        if fit:
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            self._scaler_params = ScalerParams(method='robust', median=median, mad=mad)
        else:
            if self._scaler_params is None or self._scaler_params.method != 'robust':
                raise ValueError("Scaler not fitted. Set fit=True first.")
            median = self._scaler_params.median
            mad = self._scaler_params.mad

        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)

        return (data - median) / mad

    # ========== Feature Engineering ==========

    @staticmethod
    def polynomial_features(data: np.ndarray, degree: int = 2,
                          include_bias: bool = False) -> np.ndarray:
        """
        Generate polynomial features.

        Args:
            data: Input data (2D array)
            degree: Polynomial degree
            include_bias: Include bias column (all ones)

        Returns:
            Polynomial features
        """
        n_samples, n_features = data.shape
        features = [data]

        for d in range(2, degree + 1):
            features.append(data ** d)

        result = np.hstack(features)

        if include_bias:
            bias = np.ones((n_samples, 1))
            result = np.hstack([bias, result])

        return result

    @staticmethod
    def interaction_features(data: np.ndarray,
                           feature_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Generate interaction features (pairwise products).

        Args:
            data: Input data (2D array)
            feature_pairs: List of (i, j) pairs to interact (None = all pairs)

        Returns:
            Data with interaction features appended
        """
        n_samples, n_features = data.shape

        if feature_pairs is None:
            # Generate all pairs
            feature_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

        interactions = []
        for i, j in feature_pairs:
            interactions.append((data[:, i] * data[:, j]).reshape(-1, 1))

        if interactions:
            return np.hstack([data] + interactions)
        else:
            return data

    @staticmethod
    def binning(data: np.ndarray, n_bins: int = 10,
               strategy: Literal['uniform', 'quantile'] = 'uniform') -> np.ndarray:
        """
        Bin continuous data into discrete bins.

        Args:
            data: Input data (1D array)
            n_bins: Number of bins
            strategy: Binning strategy

        Returns:
            Bin indices
        """
        if strategy == 'uniform':
            bins = np.linspace(data.min(), data.max(), n_bins + 1)
        elif strategy == 'quantile':
            bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return np.digitize(data, bins[1:-1])

    # ========== Missing Value Imputation ==========

    @staticmethod
    def impute_missing(data: np.ndarray,
                      strategy: Literal['mean', 'median', 'mode', 'forward', 'backward', 'linear'] = 'mean',
                      missing_values: float = np.nan) -> np.ndarray:
        """
        Impute missing values.

        Args:
            data: Input data
            strategy: Imputation strategy
            missing_values: Value to treat as missing

        Returns:
            Data with imputed values
        """
        result = data.copy()
        mask = np.isnan(result) if np.isnan(missing_values) else (result == missing_values)

        if strategy == 'mean':
            fill_value = np.nanmean(result)
            result[mask] = fill_value
        elif strategy == 'median':
            fill_value = np.nanmedian(result)
            result[mask] = fill_value
        elif strategy == 'mode':
            from scipy import stats
            fill_value = stats.mode(result[~mask], keepdims=True).mode[0]
            result[mask] = fill_value
        elif strategy == 'forward':
            # Forward fill
            for i in range(len(result)):
                if mask[i] and i > 0:
                    result[i] = result[i - 1]
        elif strategy == 'backward':
            # Backward fill
            for i in range(len(result) - 1, -1, -1):
                if mask[i] and i < len(result) - 1:
                    result[i] = result[i + 1]
        elif strategy == 'linear':
            # Linear interpolation
            indices = np.arange(len(result))
            valid_mask = ~mask
            if np.any(valid_mask):
                result[mask] = np.interp(indices[mask], indices[valid_mask], result[valid_mask])

        return result

    # ========== Dimensionality Reduction ==========

    @staticmethod
    def pca(data: np.ndarray, n_components: Optional[int] = None,
           variance_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Principal Component Analysis.

        Args:
            data: Input data (2D array)
            n_components: Number of components (None = auto based on variance)
            variance_threshold: Cumulative variance threshold for auto selection

        Returns:
            Tuple of (transformed_data, components, explained_variance_ratio)
        """
        # Center data
        mean = np.mean(data, axis=0)
        data_centered = data - mean

        # Compute covariance matrix
        cov_matrix = np.cov(data_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Determine number of components
        if n_components is None:
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Select components
        components = eigenvectors[:, :n_components]
        explained_var = explained_variance_ratio[:n_components]

        # Transform data
        transformed = np.dot(data_centered, components)

        logger.info(
            f"PCA: {n_components} components explain "
            f"{np.sum(explained_var):.2%} of variance"
        )

        return transformed, components, explained_var

    # ========== Feature Selection ==========

    @staticmethod
    def variance_threshold(data: np.ndarray,
                          threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove low-variance features.

        Args:
            data: Input data (2D array)
            threshold: Variance threshold

        Returns:
            Tuple of (filtered_data, selected_feature_indices)
        """
        variances = np.var(data, axis=0)
        selected = variances > threshold

        logger.info(
            f"Variance threshold: {np.sum(selected)}/{len(selected)} features selected"
        )

        return data[:, selected], np.where(selected)[0]

    @staticmethod
    def correlation_threshold(data: np.ndarray,
                             threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove highly correlated features.

        Args:
            data: Input data (2D array)
            threshold: Correlation threshold

        Returns:
            Tuple of (filtered_data, selected_feature_indices)
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)

        # Find highly correlated pairs
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        to_drop = set()

        for i in range(upper_triangle.shape[0]):
            for j in range(i + 1, upper_triangle.shape[1]):
                if upper_triangle[i, j] > threshold:
                    to_drop.add(j)

        selected = [i for i in range(data.shape[1]) if i not in to_drop]

        logger.info(
            f"Correlation threshold: {len(selected)}/{data.shape[1]} features selected"
        )

        return data[:, selected], np.array(selected)

    # ========== Encoding ==========

    @staticmethod
    def one_hot_encode(labels: np.ndarray,
                      n_classes: Optional[int] = None) -> np.ndarray:
        """
        One-hot encode labels.

        Args:
            labels: Integer labels
            n_classes: Number of classes (auto-detected if None)

        Returns:
            One-hot encoded array
        """
        if n_classes is None:
            n_classes = int(np.max(labels)) + 1

        n_samples = len(labels)
        encoded = np.zeros((n_samples, n_classes))
        encoded[np.arange(n_samples), labels.astype(int)] = 1

        return encoded

    @staticmethod
    def label_encode(categories: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Encode categorical labels as integers.

        Args:
            categories: Categorical labels

        Returns:
            Tuple of (encoded_labels, category_mapping)
        """
        unique_categories = np.unique(categories)
        mapping = {cat: i for i, cat in enumerate(unique_categories)}
        encoded = np.array([mapping[cat] for cat in categories])

        return encoded, mapping
