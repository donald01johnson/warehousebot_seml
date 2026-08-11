"""
Warehouse Robot Navigation System — Assignment II Production Package.

A production-grade refactoring of the Assignment I autonomous warehouse
robot navigation system.

Course      : AIMLCZG546
Institution : BITS Pilani WILP
Group       : 212
Members     : Donald Johnson A
"""

from .config import (
    DEFAULT_COLS,
    DEFAULT_NUM_EPISODES,
    DEFAULT_OBSTACLE_PROB,
    DEFAULT_ROWS,
    MOVES,
    TIME_PER_STEP,
)
from .entities import EpisodeLog, Point
from .grid import WarehouseGrid, neighbors
from .logging_config import get_logger, setup_logging
from .planner import bfs_shortest_path, greedy_random_path
from .simulator import RobotSimulator

__version__ = "2.0.0"
__author__ = (
    "Group 212 — Donald Johnson A"
)
__course__ = "AIMLCZG546, BITS Pilani WILP"

__all__ = [
    "EpisodeLog",
    "Point",
    "WarehouseGrid",
    "neighbors",
    "RobotSimulator",
    "bfs_shortest_path",
    "greedy_random_path",
    "setup_logging",
    "get_logger",
    "MOVES",
    "TIME_PER_STEP",
    "DEFAULT_ROWS",
    "DEFAULT_COLS",
    "DEFAULT_OBSTACLE_PROB",
    "DEFAULT_NUM_EPISODES",
]
