"""Data models for CSV processing configuration.

This module contains data classes that define the configuration
for various processing operations in the Data Processor.
"""

from dataclasses import dataclass, field
from typing import Any  # noqa: ICN003


@dataclass
class FilterConfig:
    """Configuration for signal filtering operations."""

    filter_type: str = "Moving Average"

    # Moving Average parameters
    ma_window: int = 10

    # Butterworth filter parameters
    bw_order: int = 3
    bw_cutoff: float = 0.1

    # Median filter parameters
    median_kernel: int = 5

    # Savitzky-Golay parameters
    savgol_window: int = 11
    savgol_polyorder: int = 2

    # Gaussian filter parameters
    gaussian_sigma: float = 1.0
    gaussian_mode: str = "reflect"

    # Hampel filter parameters
    hampel_window: int = 5
    hampel_threshold: float = 3.0

    # Z-Score filter parameters
    zscore_threshold: float = 3.0
    zscore_method: str = "modified"

    # FFT filter parameters
    fft_freq_low: float = 0.1
    fft_freq_high: float = 0.3
    fft_transition_bw: float = 0.05
    fft_window_shape: str = "Gaussian"
    fft_zero_phase: bool = True
    fft_freq_unit: str = "normalized"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for filter engine."""
        return {
            "filter_type": self.filter_type,
            "ma_window": self.ma_window,
            "bw_order": self.bw_order,
            "bw_cutoff": self.bw_cutoff,
            "median_kernel": self.median_kernel,
            "savgol_window": self.savgol_window,
            "savgol_polyorder": self.savgol_polyorder,
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_mode": self.gaussian_mode,
            "hampel_window": self.hampel_window,
            "hampel_threshold": self.hampel_threshold,
            "zscore_threshold": self.zscore_threshold,
            "zscore_method": self.zscore_method,
            "fft_freq_low": self.fft_freq_low,
            "fft_freq_high": self.fft_freq_high,
            "fft_transition_bw": self.fft_transition_bw,
            "fft_window_shape": self.fft_window_shape,
            "fft_zero_phase": self.fft_zero_phase,
            "fft_freq_unit": self.fft_freq_unit,
        }


@dataclass
class PlottingConfig:
    """Configuration for plotting operations."""

    selected_signals: list[str] = field(default_factory=list)
    reference_signals: list[str] = field(default_factory=list)
    plot_title: str = ""
    x_label: str = ""
    y_label: str = "Value"
    legend_position: str = "best"
    grid_enabled: bool = True
    line_width: float = 1.0

    # Time range
    use_time_range: bool = False
    start_time: str = "00:00"
    end_time: str = "23:59"

    # Plot type
    plot_type: str = "line"  # line, scatter, both


@dataclass
class ProcessingConfig:
    """Main configuration for data processing operations."""

    # File selection
    selected_files: list[str] = field(default_factory=list)

    # Signal selection
    available_signals: list[str] = field(default_factory=list)
    selected_signals: list[str] = field(default_factory=list)

    # Processing options
    apply_filtering: bool = False
    apply_integration: bool = False
    apply_differentiation: bool = False

    # Filter configuration
    filter_config: FilterConfig = field(default_factory=FilterConfig)

    # Plotting configuration
    plotting_config: PlottingConfig = field(default_factory=PlottingConfig)

    # Output options
    output_directory: str | None = None
    output_format: str = "csv"
    include_original: bool = True

    # Custom variables
    custom_variables: dict[str, str] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for signal integration."""

    signals_to_integrate: list[str] = field(default_factory=list)
    integration_method: str = "cumulative"  # cumulative, trapezoidal
    reset_on_zero: bool = False
    initial_value: float = 0.0


@dataclass
class DifferentiationConfig:
    """Configuration for signal differentiation."""

    signals_to_differentiate: list[str] = field(default_factory=list)
    differentiation_order: int = 1
    method: str = "central"  # forward, backward, central
