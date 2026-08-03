from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[int, int]


@dataclass
class EpisodeLog:
    start: Point
    goal: Point
    path: List[Point]
    shortest_path_len: int
    travel_time: float
    collisions: int
    replans: int
    obstacle_density: float
