"""Threading utilities for data processing operations.

This module provides decoupled threading utilities that use callbacks
instead of tight GUI coupling for better testability and reusability.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any, Optional


class ConversionThread(threading.Thread):
    """Thread for handling file conversion operations (decoupled from GUI).

    Uses callback functions instead of GUI references for better separation of concerns.
    """

    def __init__(
        self,
        conversion_fn: Callable[[], Any],
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Initialize the conversion thread.

        Args:
            conversion_fn: Function to perform the conversion
            on_complete: Optional callback when conversion completes successfully
            on_error: Optional callback when conversion fails
            on_progress: Optional callback for progress updates (percent, message)
        """
        super().__init__()
        self.conversion_fn = conversion_fn
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_progress = on_progress
        self.daemon = True
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Execute the conversion operation in a separate thread."""
        try:
            result = self.conversion_fn()
            if self.on_complete:
                self.on_complete(result)
        except Exception as e:
            self.logger.exception(f"Conversion error: {e}")
            if self.on_error:
                self.on_error(e)


class CombinedConversionThread(ConversionThread):
    """Thread for handling combined file conversion.

    Convenience wrapper around ConversionThread for backward compatibility.
    """

    def __init__(
        self,
        conversion_fn: Callable[[], Any],
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Initialize the combined conversion thread.

        Args:
            conversion_fn: Function to perform the combined conversion
            on_complete: Optional callback when conversion completes
            on_error: Optional callback when conversion fails
            on_progress: Optional callback for progress updates
        """
        super().__init__(
            conversion_fn=conversion_fn,
            on_complete=on_complete,
            on_error=on_error,
            on_progress=on_progress,
        )


class SeparateConversionThread(ConversionThread):
    """Thread for handling separate file conversion.

    Convenience wrapper around ConversionThread for backward compatibility.
    """

    def __init__(
        self,
        conversion_fn: Callable[[], Any],
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Initialize the separate conversion thread.

        Args:
            conversion_fn: Function to perform the separate conversion
            on_complete: Optional callback when conversion completes
            on_error: Optional callback when conversion fails
            on_progress: Optional callback for progress updates
        """
        super().__init__(
            conversion_fn=conversion_fn,
            on_complete=on_complete,
            on_error=on_error,
            on_progress=on_progress,
        )
