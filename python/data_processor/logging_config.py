"""Centralized logging configuration for the Data Processor application.

This module provides standardized logging setup and utilities for consistent
logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOGGING_INITIALIZED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    console_output: bool = True,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Format string for log messages
        console_output: Whether to output logs to console
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def ensure_logging_initialized() -> None:
    """Initialize logging once in a lazy manner."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    # Respect existing handlers if user already configured logging
    if logging.getLogger().handlers:
        _LOGGING_INITIALIZED = True
        return

    setup_logging(
        level=logging.INFO,
        console_output=True,
        log_format=DEFAULT_LOG_FORMAT,
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    ensure_logging_initialized()
    return logging.getLogger(name)


class LoggerAdapter:
    """Adapter to make legacy callback-based logging compatible with Python logging.

    This allows gradual migration from callback-based logging to standard logging.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the logger adapter.

        Args:
            logger: Optional logger instance. If None, creates default logger.
        """
        self.logger = logger or get_logger(__name__)

    def __call__(self, message: str, level: int = logging.INFO) -> None:
        """Allow adapter to be called like a function (for callback compatibility).

        Args:
            message: Log message
            level: Logging level
        """
        self.logger.log(level, message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message.

        Args:
            message: Error message
            exc_info: Whether to include exception traceback
        """
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log critical message.

        Args:
            message: Critical message
            exc_info: Whether to include exception traceback
        """
        self.logger.critical(message, exc_info=exc_info)


def init_default_logging(force: bool = False) -> None:
    """Explicit hook to configure logging for entry points."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED and not force:
        return
    setup_logging(
        level=logging.INFO,
        console_output=True,
        log_format=DEFAULT_LOG_FORMAT,
    )
    _LOGGING_INITIALIZED = True
