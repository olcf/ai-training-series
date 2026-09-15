"""Logging configuration for RHAPSODY.

This module provides simple logging configuration utilities.

Usage:
    import logging
    import rhapsody

    # Enable debug logging
    rhapsody.enable_logging(logging.DEBUG)

    # Or with custom format
    rhapsody.enable_logging(
        level=logging.DEBUG,
        format_string='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

Note:
    Call enable_logging() BEFORE creating any backends to ensure all logs are captured.
"""

import logging
import sys
from typing import Any
from typing import Optional


def enable_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = "%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
    stream: Optional[Any] = None,
    force: bool = True,
) -> None:
    """Enable logging for RHAPSODY.

    This function configures Python's logging system for RHAPSODY. It should be
    called BEFORE creating any backends to ensure all logs are captured properly.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string. If None, uses a default format
        stream: Output stream (default: sys.stderr)
        force: If True, reconfigure even if logging was already configured
               (Python 3.8+). This is important when Dragon or other frameworks
               have already configured logging.

    Example:
        ::

            import logging
            import rhapsody
            rhapsody.enable_logging(logging.DEBUG)
            backend = await rhapsody.get_backend("dragon")
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s"

    if stream is None:
        stream = sys.stderr

    # For Python 3.8+, use force=True to reconfigure existing loggers
    # This is critical for Dragon which configures logging at runtime
    try:
        logging.basicConfig(
            level=level,
            format=format_string,
            stream=stream,
            force=force,  # Python 3.8+
        )
    except TypeError:
        # Python < 3.8 doesn't support 'force' parameter
        # Fall back to manual reconfiguration
        root = logging.getLogger()
        root.setLevel(level)

        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # Add new handler
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(format_string))
        root.addHandler(handler)

    # Explicitly configure the rhapsody logger hierarchy
    # This ensures rhapsody loggers pick up the configuration even if
    # other frameworks (like Dragon) reconfigure the root logger
    rhapsody_logger = logging.getLogger("rhapsody")
    rhapsody_logger.setLevel(level)

    # Add a handler directly to the rhapsody logger to ensure logs are emitted
    # even if the root logger's handlers are changed by other frameworks
    # First, remove any existing handlers to avoid duplicates
    rhapsody_logger.handlers.clear()

    # Add our handler
    rhapsody_handler = logging.StreamHandler(stream)
    rhapsody_handler.setLevel(level)
    rhapsody_handler.setFormatter(logging.Formatter(format_string))
    rhapsody_logger.addHandler(rhapsody_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    rhapsody_logger.propagate = False

    # Also configure any already-created rhapsody child loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("rhapsody"):
            existing_logger = logging.getLogger(logger_name)
            existing_logger.setLevel(level)
