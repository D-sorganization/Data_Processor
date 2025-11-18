"""
Configuration loader for Data Processor Pro.

Supports loading configuration from YAML/TOML files with validation
and environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .models import AppConfig, ProcessingConfig, VisualizationConfig, PerformanceConfig, UIConfig


class ConfigLoader:
    """Load and save application configuration."""

    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Environment variable substitution
        config_dict = ConfigLoader._substitute_env_vars(config_dict)

        return config_dict

    @staticmethod
    def save_yaml(config: AppConfig, file_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = asdict(config)

        # Convert Path objects to strings
        config_dict = ConfigLoader._paths_to_strings(config_dict)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

    @staticmethod
    def from_yaml(file_path: Path) -> AppConfig:
        """Create AppConfig from YAML file."""
        config_dict = ConfigLoader.load_yaml(file_path)

        # Parse nested configurations
        app_config = AppConfig(
            processing=ConfigLoader._parse_processing(config_dict.get('processing', {})),
            visualization=ConfigLoader._parse_visualization(config_dict.get('visualization', {})),
            performance=ConfigLoader._parse_performance(config_dict.get('performance', {})),
            ui=ConfigLoader._parse_ui(config_dict.get('ui', {})),
            version=config_dict.get('version', '1.0.0'),
            debug_mode=config_dict.get('debug_mode', False),
            log_level=config_dict.get('log_level', 'INFO'),
            log_file=Path(config_dict['log_file']) if config_dict.get('log_file') else None,
            session_file=Path(config_dict['session_file']) if config_dict.get('session_file') else None,
            auto_load_session=config_dict.get('auto_load_session', True),
            auto_save_session=config_dict.get('auto_save_session', True),
            plugin_dir=Path(config_dict.get('plugin_dir', Path.home() / '.data_processor_pro' / 'plugins')),
            enabled_plugins=config_dict.get('enabled_plugins', []),
        )

        # Validate
        app_config.validate()

        return app_config

    @staticmethod
    def get_default_config() -> AppConfig:
        """Get default application configuration."""
        return AppConfig()

    @staticmethod
    def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in config."""
        if isinstance(config_dict, dict):
            return {k: ConfigLoader._substitute_env_vars(v) for k, v in config_dict.items()}
        elif isinstance(config_dict, list):
            return [ConfigLoader._substitute_env_vars(item) for item in config_dict]
        elif isinstance(config_dict, str) and config_dict.startswith('${') and config_dict.endswith('}'):
            env_var = config_dict[2:-1]
            return os.environ.get(env_var, config_dict)
        else:
            return config_dict

    @staticmethod
    def _paths_to_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Path objects to strings recursively."""
        if isinstance(config_dict, dict):
            return {k: ConfigLoader._paths_to_strings(v) for k, v in config_dict.items()}
        elif isinstance(config_dict, list):
            return [ConfigLoader._paths_to_strings(item) for item in config_dict]
        elif isinstance(config_dict, Path):
            return str(config_dict)
        else:
            return config_dict

    @staticmethod
    def _parse_processing(data: Dict[str, Any]) -> ProcessingConfig:
        """Parse processing configuration."""
        from .models import FilterConfig, FilterType, IntegrationMethod, DifferentiationMethod, ResamplingMethod, AnomalyMethod

        filter_data = data.get('filter_config', {})
        filter_config = FilterConfig(
            filter_type=FilterType(filter_data.get('filter_type', 'moving_average')),
            window_size=filter_data.get('window_size', 5),
            order=filter_data.get('order', 4),
            cutoff_freq=filter_data.get('cutoff_freq', 0.1),
            low_freq=filter_data.get('low_freq'),
            high_freq=filter_data.get('high_freq'),
            kernel_size=filter_data.get('kernel_size', 5),
            sg_window=filter_data.get('sg_window', 11),
            sg_polyorder=filter_data.get('sg_polyorder', 3),
            sigma=filter_data.get('sigma', 1.0),
            gaussian_mode=filter_data.get('gaussian_mode', 'reflect'),
            hampel_window=filter_data.get('hampel_window', 5),
            hampel_threshold=filter_data.get('hampel_threshold', 3.0),
            z_threshold=filter_data.get('z_threshold', 3.0),
            z_method=filter_data.get('z_method', 'modified'),
        )

        return ProcessingConfig(
            input_files=[Path(f) for f in data.get('input_files', [])],
            output_dir=Path(data['output_dir']) if data.get('output_dir') else None,
            output_format=data.get('output_format', 'parquet'),
            signals=data.get('signals', []),
            time_column=data.get('time_column'),
            apply_filter=data.get('apply_filter', False),
            filter_config=filter_config,
            apply_integration=data.get('apply_integration', False),
            integration_method=IntegrationMethod(data.get('integration_method', 'trapezoidal')),
            apply_differentiation=data.get('apply_differentiation', False),
            differentiation_method=DifferentiationMethod(data.get('differentiation_method', 'central')),
            differentiation_order=data.get('differentiation_order', 1),
            apply_resampling=data.get('apply_resampling', False),
            resampling_rate=data.get('resampling_rate'),
            resampling_method=ResamplingMethod(data.get('resampling_method', 'linear')),
            custom_formula=data.get('custom_formula'),
            custom_variables=data.get('custom_variables', {}),
            remove_outliers=data.get('remove_outliers', False),
            outlier_method=AnomalyMethod(data.get('outlier_method', 'iqr')),
            normalize=data.get('normalize', False),
            normalization_method=data.get('normalization_method', 'zscore'),
            fill_missing=data.get('fill_missing', False),
            missing_strategy=data.get('missing_strategy', 'linear'),
        )

    @staticmethod
    def _parse_visualization(data: Dict[str, Any]) -> VisualizationConfig:
        """Parse visualization configuration."""
        return VisualizationConfig(
            plot_type=data.get('plot_type', 'line'),
            x_signal=data.get('x_signal'),
            y_signals=data.get('y_signals', []),
            z_signal=data.get('z_signal'),
            title=data.get('title', 'Signal Plot'),
            x_label=data.get('x_label', 'Time'),
            y_label=data.get('y_label', 'Value'),
            show_legend=data.get('show_legend', True),
            show_grid=data.get('show_grid', True),
            color_scheme=data.get('color_scheme', 'plotly'),
            template=data.get('template', 'plotly_dark'),
            enable_zoom=data.get('enable_zoom', True),
            enable_pan=data.get('enable_pan', True),
            enable_hover=data.get('enable_hover', True),
            enable_rangeslider=data.get('enable_rangeslider', False),
            export_width=data.get('export_width', 1920),
            export_height=data.get('export_height', 1080),
            export_format=data.get('export_format', 'html'),
        )

    @staticmethod
    def _parse_performance(data: Dict[str, Any]) -> PerformanceConfig:
        """Parse performance configuration."""
        return PerformanceConfig(
            max_workers=data.get('max_workers', 4),
            use_multiprocessing=data.get('use_multiprocessing', True),
            chunk_size=data.get('chunk_size', 10000),
            enable_cache=data.get('enable_cache', True),
            cache_dir=Path(data.get('cache_dir', Path.home() / '.cache' / 'data_processor_pro')),
            cache_size_limit_mb=data.get('cache_size_limit_mb', 1000),
            lazy_loading=data.get('lazy_loading', True),
            max_memory_mb=data.get('max_memory_mb', 4096),
            use_gpu=data.get('use_gpu', False),
            gpu_device=data.get('gpu_device', 0),
            use_numba=data.get('use_numba', True),
            numba_cache=data.get('numba_cache', True),
            use_polars=data.get('use_polars', True),
        )

    @staticmethod
    def _parse_ui(data: Dict[str, Any]) -> UIConfig:
        """Parse UI configuration."""
        from .models import ThemeMode

        return UIConfig(
            theme=ThemeMode(data.get('theme', 'dark')),
            accent_color=data.get('accent_color', '#1f77b4'),
            window_width=data.get('window_width', 1600),
            window_height=data.get('window_height', 900),
            window_title=data.get('window_title', 'Data Processor Pro'),
            sidebar_width=data.get('sidebar_width', 300),
            show_statusbar=data.get('show_statusbar', True),
            show_toolbar=data.get('show_toolbar', True),
            font_family=data.get('font_family', 'Segoe UI'),
            font_size=data.get('font_size', 10),
            code_font_family=data.get('code_font_family', 'Consolas'),
            enable_drag_drop=data.get('enable_drag_drop', True),
            enable_keyboard_shortcuts=data.get('enable_keyboard_shortcuts', True),
            autosave_interval_seconds=data.get('autosave_interval_seconds', 300),
            max_recent_files=data.get('max_recent_files', 10),
            recent_files=[Path(f) for f in data.get('recent_files', [])],
        )
