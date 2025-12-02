"""Core signal processing operations.

This module provides the business logic for signal processing operations,
decoupled from the GUI layer for better testability and reusability.
"""

import numpy as np
import pandas as pd

from data_processor.logging_config import get_logger
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)
from data_processor.vectorized_filter_engine import VectorizedFilterEngine

logger = get_logger(__name__)


class SignalProcessor:
    """Core signal processing operations."""

    def __init__(self) -> None:
        """Initialize the signal processor."""
        self.filter_engine = VectorizedFilterEngine()
        self.logger = logger

    def apply_filter(
        self,
        df: pd.DataFrame,
        filter_config: FilterConfig,
        signals: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply filter to signals in DataFrame.

        Args:
            df: DataFrame containing signals
            filter_config: Filter configuration
            signals: List of signal names to filter (None = all numeric columns)

        Returns:
            DataFrame with filtered signals
        """
        if signals is None:
            signals = df.select_dtypes(include=np.number).columns.tolist()

        logger.info(
            f"Applying {filter_config.filter_type} filter to {len(signals)} signals",
        )

        # Convert config to parameters dict
        params = filter_config.to_dict()

        # Apply filter using vectorized engine
        return self.filter_engine.apply_filter_batch(
            df=df,
            filter_type=filter_config.filter_type,
            params=params,
            signal_names=signals,
        )
    def integrate_signals(
        self,
        df: pd.DataFrame,
        config: IntegrationConfig,
    ) -> pd.DataFrame:
        """Integrate selected signals.

        Args:
            df: DataFrame containing signals
            config: Integration configuration

        Returns:
            DataFrame with integrated signals added
        """
        result_df = df.copy()

        for signal in config.signals_to_integrate:
            if signal not in df.columns:
                logger.warning(f"Signal {signal} not found, skipping integration")
                continue

            logger.info(f"Integrating signal: {signal}")

            if config.integration_method == "cumulative":
                integrated = df[signal].cumsum() + config.initial_value
            elif config.integration_method == "trapezoidal":
                # Trapezoidal integration with proper time step handling
                values = df[signal].values

                # Calculate time differences (dt)
                if isinstance(df.index, pd.DatetimeIndex):
                    # Convert datetime differences to seconds
                    dt = df.index.to_series().diff().dt.total_seconds().values[1:]
                elif pd.api.types.is_numeric_dtype(df.index):
                    # Numeric index - calculate differences
                    dt = np.diff(df.index.values)
                else:
                    # Fallback to uniform dt=1 if index is not datetime or numeric
                    logger.warning(
                        f"Non-numeric index detected for {signal}, assuming dt=1",
                    )
                    dt = np.ones(len(values) - 1)

                # Trapezoidal rule: integral = sum((y[i] + y[i+1]) / 2 * dt[i])
                trapezoids = (values[:-1] + values[1:]) / 2 * dt
                integrated = pd.Series(
                    np.cumsum(trapezoids),
                    index=df.index[1:],
                )
                # Add initial value
                integrated = pd.concat(
                    [
                        pd.Series([config.initial_value], index=[df.index[0]]),
                        integrated,
                    ],
                )
            else:
                logger.error(f"Unknown integration method: {config.integration_method}")
                continue

            # Add as new column
            result_df[f"{signal}_integrated"] = integrated

        return result_df

    def differentiate_signals(
        self,
        df: pd.DataFrame,
        config: DifferentiationConfig,
    ) -> pd.DataFrame:
        """Differentiate selected signals.

        Args:
            df: DataFrame containing signals
            config: Differentiation configuration

        Returns:
            DataFrame with differentiated signals added
        """
        result_df = df.copy()

        for signal in config.signals_to_differentiate:
            if signal not in df.columns:
                logger.warning(f"Signal {signal} not found, skipping differentiation")
                continue

            logger.info(
                f"Differentiating signal: {signal} (order={config.differentiation_order})",
            )

            # Get signal values
            values = df[signal].values

            # Apply differentiation based on method
            if config.method == "forward":
                deriv = np.diff(
                    values,
                    n=config.differentiation_order,
                    prepend=[values[0]] * config.differentiation_order,
                )
            elif config.method == "backward":
                deriv = np.diff(
                    values,
                    n=config.differentiation_order,
                    append=[values[-1]] * config.differentiation_order,
                )
            elif config.method == "central":
                # Central difference
                deriv = np.gradient(values, edge_order=2)
                # Apply multiple times for higher orders
                for _ in range(config.differentiation_order - 1):
                    deriv = np.gradient(deriv, edge_order=2)
            else:
                logger.error(f"Unknown differentiation method: {config.method}")
                continue

            # Add as new column
            suffix = (
                f"_deriv{config.differentiation_order}"
                if config.differentiation_order > 1
                else "_deriv"
            )
            result_df[f"{signal}{suffix}"] = deriv

        return result_df

    def apply_custom_formula(
        self,
        df: pd.DataFrame,
        formula_name: str,
        formula: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Apply custom formula to create new signal.

        Args:
            df: DataFrame containing signals
            formula_name: Name for the new signal
            formula: Formula expression (can reference signal names)

        Returns:
            Tuple of (updated DataFrame, success flag)
        """
        try:
            logger.info(f"Applying custom formula: {formula_name} = {formula}")

            # Evaluate formula in DataFrame context
            # This allows formulas like "signal1 + signal2" to work
            result = df.eval(formula)

            # Add as new column
            df[formula_name] = result

            logger.info(f"Successfully created signal: {formula_name}")
            return df, True

        except Exception as e:
            logger.error(f"Error applying formula '{formula}': {e}", exc_info=True)
            return df, False

    def detect_signal_statistics(self, df: pd.DataFrame, signal: str) -> dict:
        """Calculate statistics for a signal.

        Args:
            df: DataFrame containing the signal
            signal: Signal name

        Returns:
            Dictionary with statistics
        """
        if signal not in df.columns:
            return {"error": f"Signal {signal} not found"}

        data = df[signal].dropna()

        return {
            "count": len(data),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "median": float(data.median()),
            "q25": float(data.quantile(0.25)),
            "q75": float(data.quantile(0.75)),
        }
    def resample_signals(
        self,
        df: pd.DataFrame,
        target_sampling_rate: str,
        signals: list[str] | None = None,
    ) -> pd.DataFrame:
        """Resample signals to a target sampling rate.

        Args:
            df: DataFrame with DatetimeIndex
            target_sampling_rate: Target sampling rate (e.g., '1S', '100ms')
            signals: Signals to resample (None = all)

        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame must have DatetimeIndex for resampling")
            return df

        if signals is None:
            signals = df.columns.tolist()

        logger.info(f"Resampling {len(signals)} signals to {target_sampling_rate}")

        # Resample using mean interpolation
        resampled = df[signals].resample(target_sampling_rate).mean()

        # Interpolate NaN values
        return resampled.interpolate(method="linear")



# Convenience function for backward compatibility
def apply_filter_to_signals(
    df: pd.DataFrame,
    filter_type: str,
    filter_params: dict,
    signals: list[str] | None = None,
) -> pd.DataFrame:
    """Apply filter to signals (convenience function).

    Args:
        df: DataFrame containing signals
        filter_type: Type of filter
        filter_params: Filter parameters
        signals: Signals to filter

    Returns:
        Filtered DataFrame
    """
    processor = SignalProcessor()

    # Create config from parameters
    config = FilterConfig(filter_type=filter_type, **filter_params)

    return processor.apply_filter(df, config, signals)
