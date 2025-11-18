"""
Type-safe configuration models for Data Processor Pro.

Provides comprehensive, validated configuration dataclasses for all aspects
of the application including processing, visualization, performance, and UI.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum


class ThemeMode(str, Enum):
    """UI theme modes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class FilterType(str, Enum):
    """Available signal filter types."""
    MOVING_AVERAGE = "moving_average"
    BUTTERWORTH_LOWPASS = "butterworth_lowpass"
    BUTTERWORTH_HIGHPASS = "butterworth_highpass"
    BUTTERWORTH_BANDPASS = "butterworth_bandpass"
    MEDIAN = "median"
    SAVITZKY_GOLAY = "savitzky_golay"
    GAUSSIAN = "gaussian"
    HAMPEL = "hampel"
    Z_SCORE = "z_score"
    FFT_LOWPASS = "fft_lowpass"
    FFT_HIGHPASS = "fft_highpass"
    FFT_BANDPASS = "fft_bandpass"
    FFT_BANDSTOP = "fft_bandstop"
    KALMAN = "kalman"
    BILATERAL = "bilateral"


class IntegrationMethod(str, Enum):
    """Integration calculation methods."""
    CUMULATIVE = "cumulative"
    TRAPEZOIDAL = "trapezoidal"
    SIMPSON = "simpson"


class DifferentiationMethod(str, Enum):
    """Differentiation calculation methods."""
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


class ResamplingMethod(str, Enum):
    """Time series resampling methods."""
    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"
    PCHIP = "pchip"


class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    DBSCAN = "dbscan"
    Z_SCORE = "z_score"
    IQR = "iqr"


@dataclass
class FilterConfig:
    """Configuration for signal filtering operations."""

    # General
    filter_type: FilterType = FilterType.MOVING_AVERAGE

    # Moving Average
    window_size: int = 5

    # Butterworth
    order: int = 4
    cutoff_freq: float = 0.1
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None

    # Median
    kernel_size: int = 5

    # Savitzky-Golay
    sg_window: int = 11
    sg_polyorder: int = 3

    # Gaussian
    sigma: float = 1.0
    gaussian_mode: str = "reflect"

    # Hampel
    hampel_window: int = 5
    hampel_threshold: float = 3.0

    # Z-Score
    z_threshold: float = 3.0
    z_method: Literal["modified", "standard"] = "modified"

    # FFT
    fft_transition_width: float = 0.1
    fft_window: str = "hamming"
    fft_zero_phase: bool = True

    # Kalman Filter (new)
    process_variance: float = 1e-5
    measurement_variance: float = 1e-2

    # Bilateral Filter (new)
    bilateral_sigma_spatial: float = 1.0
    bilateral_sigma_intensity: float = 1.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if not 0 < self.cutoff_freq < 0.5:
            raise ValueError("cutoff_freq must be in (0, 0.5)")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""

    # Input/Output
    input_files: List[Path] = field(default_factory=list)
    output_dir: Optional[Path] = None
    output_format: str = "parquet"

    # Signal Selection
    signals: List[str] = field(default_factory=list)
    time_column: Optional[str] = None

    # Operations
    apply_filter: bool = False
    filter_config: FilterConfig = field(default_factory=FilterConfig)

    apply_integration: bool = False
    integration_method: IntegrationMethod = IntegrationMethod.TRAPEZOIDAL

    apply_differentiation: bool = False
    differentiation_method: DifferentiationMethod = DifferentiationMethod.CENTRAL
    differentiation_order: int = 1

    apply_resampling: bool = False
    resampling_rate: Optional[float] = None
    resampling_method: ResamplingMethod = ResamplingMethod.LINEAR

    # Custom Formula
    custom_formula: Optional[str] = None
    custom_variables: Dict[str, Any] = field(default_factory=dict)

    # Advanced Options
    remove_outliers: bool = False
    outlier_method: AnomalyMethod = AnomalyMethod.IQR

    normalize: bool = False
    normalization_method: Literal["minmax", "zscore", "robust"] = "zscore"

    fill_missing: bool = False
    missing_strategy: Literal["forward", "backward", "linear", "mean", "median"] = "linear"


@dataclass
class VisualizationConfig:
    """Configuration for data visualization."""

    # Plot Type
    plot_type: Literal["line", "scatter", "bar", "area", "box", "violin", "heatmap", "3d"] = "line"

    # Signals
    x_signal: Optional[str] = None
    y_signals: List[str] = field(default_factory=list)
    z_signal: Optional[str] = None  # For 3D plots

    # Styling
    title: str = "Signal Plot"
    x_label: str = "Time"
    y_label: str = "Value"
    show_legend: bool = True
    show_grid: bool = True

    # Colors and Theme
    color_scheme: str = "plotly"
    template: str = "plotly_dark"

    # Interactive Features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    enable_rangeslider: bool = False

    # Export Options
    export_width: int = 1920
    export_height: int = 1080
    export_format: Literal["png", "svg", "pdf", "html"] = "html"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Parallel Processing
    max_workers: int = 4
    use_multiprocessing: bool = True
    chunk_size: int = 10000

    # Caching
    enable_cache: bool = True
    cache_dir: Path = Path.home() / ".cache" / "data_processor_pro"
    cache_size_limit_mb: int = 1000

    # Memory Management
    lazy_loading: bool = True
    max_memory_mb: int = 4096

    # GPU Acceleration (if available)
    use_gpu: bool = False
    gpu_device: int = 0

    # JIT Compilation
    use_numba: bool = True
    numba_cache: bool = True

    # Database
    use_polars: bool = True  # Use Polars instead of Pandas for better performance


@dataclass
class UIConfig:
    """Configuration for user interface."""

    # Theme
    theme: ThemeMode = ThemeMode.DARK
    accent_color: str = "#1f77b4"

    # Window
    window_width: int = 1600
    window_height: int = 900
    window_title: str = "Data Processor Pro"

    # Layout
    sidebar_width: int = 300
    show_statusbar: bool = True
    show_toolbar: bool = True

    # Font
    font_family: str = "Segoe UI"
    font_size: int = 10
    code_font_family: str = "Consolas"

    # Features
    enable_drag_drop: bool = True
    enable_keyboard_shortcuts: bool = True
    autosave_interval_seconds: int = 300

    # Recent Files
    max_recent_files: int = 10
    recent_files: List[Path] = field(default_factory=list)


@dataclass
class AppConfig:
    """Main application configuration."""

    # Sub-configurations
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Application Settings
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[Path] = None

    # Session
    session_file: Optional[Path] = None
    auto_load_session: bool = True
    auto_save_session: bool = True

    # Plugins
    plugin_dir: Path = Path.home() / ".data_processor_pro" / "plugins"
    enabled_plugins: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate entire configuration."""
        self.processing.filter_config.validate()

        if self.performance.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if self.performance.max_memory_mb < 512:
            raise ValueError("max_memory_mb must be >= 512")

        if self.ui.window_width < 800 or self.ui.window_height < 600:
            raise ValueError("Window dimensions must be at least 800x600")
