"""Threading utilities for data processing operations."""

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from converter_tab import ConverterTab


class ConversionThread(threading.Thread):
    """Thread for handling file conversion operations."""

    def __init__(self, converter_tab: "ConverterTab", conversion_type: str) -> None:
        """Initialize the conversion thread.

        Args:
            converter_tab: The converter tab instance
            conversion_type: Type of conversion to perform

        """
        super().__init__()
        self.converter_tab = converter_tab
        self.conversion_type = conversion_type
        self.daemon = True

    def run(self) -> None:
        """Execute the conversion operation in a separate thread."""
        try:
            if self.conversion_type in {"combined", "separate"}:
                self.converter_tab._perform_conversion()
        except Exception as e:
            logging.exception(f"Conversion error: {e}")


class CombinedConversionThread(threading.Thread):
    """Thread for handling combined file conversion."""

    def __init__(self, converter_tab: "ConverterTab") -> None:
        """Initialize the combined conversion thread.

        Args:
            converter_tab: The converter tab instance

        """
        super().__init__()
        self.converter_tab = converter_tab
        self.daemon = True

    def convert_combined_files(self) -> None:
        """Convert files in combined mode."""
        try:
            self.converter_tab._perform_conversion()
        except Exception as e:
            logging.exception(f"Combined conversion error: {e}")

    def run(self) -> None:
        """Execute the combined conversion operation."""
        self.convert_combined_files()


class SeparateConversionThread(threading.Thread):
    """Thread for handling separate file conversion."""

    def __init__(self, converter_tab: "ConverterTab") -> None:
        """Initialize the separate conversion thread.

        Args:
            converter_tab: The converter tab instance

        """
        super().__init__()
        self.converter_tab = converter_tab
        self.daemon = True

    def convert_separate_files(self) -> None:
        """Convert files in separate mode."""
        try:
            self.converter_tab._perform_conversion()
        except Exception as e:
            logging.exception(f"Separate conversion error: {e}")

    def run(self) -> None:
        """Execute the separate conversion operation."""
        self.convert_separate_files()
