"""
Data Processor Pro - Professional-Grade Data Analysis Platform
================================================================

A high-performance, feature-rich data processing application designed for
professional data analysis enthusiasts and engineers.

Features:
    - Ultra-fast processing with Polars and Numba JIT compilation
    - 15+ advanced signal processing algorithms
    - Professional analytics (ML, statistics, anomaly detection)
    - Interactive visualizations with Plotly
    - Modern UI with dark/light themes
    - Session management and undo/redo
    - Drag-and-drop file support
    - 20+ file format support
    - Plugin architecture for extensibility

Author: Data Processor Pro Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Data Processor Pro Team"

from .core.engine import DataEngine
from .core.processor import SignalProcessor
from .analytics.statistics import StatisticalAnalyzer
from .analytics.ml import MLPreprocessor
from .analytics.anomaly import AnomalyDetector

__all__ = [
    "DataEngine",
    "SignalProcessor",
    "StatisticalAnalyzer",
    "MLPreprocessor",
    "AnomalyDetector",
]
