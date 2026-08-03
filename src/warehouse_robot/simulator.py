from typing import Tuple

from .entities import Point
from .grid import WarehouseGrid


class RobotSimulator:

    def __init__(
        self,
        grid: WarehouseGrid,
        start: Point,
    ):
        self.grid = grid
        self.pos = start

    def step(
        self,
        next_pos: Point,
    ) -> Tuple[Point, bool]:

        if not self.grid.is_free(next_pos):

            return self.pos, True

        self.pos = next_pos

        return self.pos, False
