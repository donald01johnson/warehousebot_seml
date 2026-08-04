"""
Centralized logging configuration for the Warehouse Robot Navigation System.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 101
"""

import logging
import os
import sys
from datetime import datetime

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    enable_file_logging: bool = True,
) -> logging.Logger:
    """
    Set up application-wide logging with console and optional file output.

    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir: Directory where log files will be written.
        enable_file_logging: If True, writes logs to a timestamped file.

    Returns:
        Root logger for the warehouse_robot package.
    """
    os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]

    if enable_file_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"warehouse_robot_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("warehouse_robot")
    logger.info(
        "Logging initialized at level: %s", logging.getLevelName(log_level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named child logger under the warehouse_robot namespace.

    Args:
        name: Module name (e.g., 'grid', 'planner').

    Returns:
        Named logger instance.
    """
    return logging.getLogger(f"warehouse_robot.{name}")
