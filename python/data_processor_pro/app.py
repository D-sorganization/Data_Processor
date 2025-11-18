"""
Data Processor Pro - Main Application Entry Point

Professional-grade data processing and analysis platform.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from .config import AppConfig, ConfigLoader
from .ui import MainWindow


def setup_logging(config: AppConfig) -> None:
    """
    Configure logging.

    Args:
        config: Application configuration
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, config.log_level)

    handlers = [logging.StreamHandler()]

    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Data Processor Pro starting...")
    logger.info(f"Version: {config.version}")
    logger.info(f"Log level: {config.log_level}")
    logger.info("="*60)


def load_config(config_file: Optional[Path] = None) -> AppConfig:
    """
    Load application configuration.

    Args:
        config_file: Path to config file (uses default if None)

    Returns:
        AppConfig instance
    """
    if config_file and config_file.exists():
        try:
            config = ConfigLoader.from_yaml(config_file)
            logging.info(f"Loaded configuration from: {config_file}")
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}. Using defaults.")
            config = ConfigLoader.get_default_config()
    else:
        config = ConfigLoader.get_default_config()
        logging.info("Using default configuration")

    return config


def main(config_file: Optional[Path] = None) -> int:
    """
    Main application entry point.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Exit code
    """
    try:
        # Load configuration
        config = load_config(config_file)

        # Setup logging
        setup_logging(config)

        # Create and run main window
        app = MainWindow(config)

        # Load session if auto-load enabled
        if config.auto_load_session and config.session_file:
            if config.session_file.exists():
                logging.info(f"Auto-loading session: {config.session_file}")
                # Session loading logic would go here

        # Run application
        app.mainloop()

        logging.info("Data Processor Pro shutting down...")
        return 0

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130

    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 1


def run():
    """Convenience function to run the application."""
    sys.exit(main())


if __name__ == "__main__":
    run()
