"""
This module provides a Loggable class for configuring and managing loggers.

The Loggable class offers functionality to:
- Initialize and configure loggers with customizable settings
- Add console and file handlers to loggers
- Set log levels and formats

Key components:
- Loggable class for easy logger configuration
- Methods for adding console and file handlers
- Customizable log format and directory

Usage:
    class MyClass(Loggable):
        def __init__(self):
            super().__init__(name="MyClass", level=logging.INFO)
        
        def my_method(self):
            self.logger.info("This is a log message")

Dependencies:
- logging
- datetime
- pathlib
"""

import datetime
import logging
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class Loggable:
    """
    A base class for adding logging capabilities to other classes.

    This class provides methods to configure and manage loggers with
    customizable settings for log level, format, and output destinations.
    """

    def __init__(
        self,
        name: str = __name__,
        level: int = logging.DEBUG,
        log_format: str = DEFAULT_LOG_FORMAT,
        logs_dir: Optional[Path] = None,
    ):
        """
        Initialize a Loggable instance with a configured logger.

        Args:
            name (str): The name of the logger. Defaults to the module name.
            level (int): The logging level. Defaults to logging.DEBUG.
            log_format (str): The format string for log messages. Defaults to DEFAULT_LOG_FORMAT.
            logs_dir (Optional[Path]): The directory to store log files. If None, file logging is disabled.
        """
        self.logger = self._configure_logger(name, level, log_format, logs_dir)

    def _configure_logger(
        self, name: str, level: int, log_format: str, logs_dir: Optional[Path]
    ) -> logging.Logger:
        """
        Configure a logger with the specified settings.

        Args:
            name (str): The name of the logger.
            level (int): The logging level.
            log_format (str): The format string for log messages.
            logs_dir (Optional[Path]): The directory to store log files.

        Returns:
            logging.Logger: A configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = logging.Formatter(log_format)

        self._add_console_handler(logger, formatter)
        if logs_dir:
            self._add_file_handler(logger, formatter, logs_dir)

        return logger

    @staticmethod
    def _add_console_handler(
        logger: logging.Logger, formatter: logging.Formatter
    ) -> None:
        """
        Add a console handler to the logger.

        Args:
            logger (logging.Logger): The logger to add the handler to.
            formatter (logging.Formatter): The formatter for log messages.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    @staticmethod
    def _add_file_handler(
        logger: logging.Logger, formatter: logging.Formatter, logs_dir: Path
    ) -> None:
        """
        Add a file handler to the logger.

        Args:
            logger (logging.Logger): The logger to add the handler to.
            formatter (logging.Formatter): The formatter for log messages.
            logs_dir (Path): The directory to store log files.
        """
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"log_{datetime.date.today()}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
