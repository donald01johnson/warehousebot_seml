"""
Core data entities for the Warehouse Robot Navigation System.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 101
"""

from dataclasses import dataclass
from typing import List, Tuple

# Type alias for a 2D grid coordinate (row, col)
Point = Tuple[int, int]


@dataclass
class EpisodeLog:
    """
    Records the complete outcome of a single robot navigation episode.

    Attributes:
        start           : Starting grid coordinate of the robot.
        goal            : Target grid coordinate for the robot.
        path            : Ordered sequence of coordinates traversed.
        shortest_path_len: Length of the BFS-optimal path.
        travel_time     : Total simulated travel time (seconds).
        collisions      : Number of collision events during traversal.
        replans         : Number of times the path was replanned mid-episode.
        obstacle_density: Fraction of grid cells that are obstacles.
    """

    start: Point
    goal: Point
    path: List[Point]
    shortest_path_len: int
    travel_time: float
    collisions: int
    replans: int
    obstacle_density: float
