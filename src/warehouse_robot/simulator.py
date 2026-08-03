"""
Robot simulator for warehouse navigation.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 101
"""

from typing import Tuple

from .entities import Point
from .grid import WarehouseGrid
from .logging_config import get_logger

logger = get_logger("simulator")


class RobotSimulator:
    """
    Simulates a physical robot moving step-by-step through the warehouse grid.

    Movement rules:
        - If the next cell is free, the robot moves there.
        - If the next cell is blocked (obstacle / wall), a collision is
          recorded and the robot stays at its current position.
    """

    def __init__(self, grid: WarehouseGrid, start: Point):
        """
        Initialise the robot at a given start position.

        Args:
            grid : The warehouse grid environment.
            start: Initial grid coordinate of the robot.

        Raises:
            ValueError: If the start position is not a free cell.
        """
        if not grid.is_free(start):
            logger.error(
                "Robot start position is blocked or out of bounds: %s", start)
            raise ValueError(f"Start position {start} is not a"
                             f"free cell in the grid.")

        self.grid = grid
        self.pos = start
        logger.info("RobotSimulator initialised at position: %s", start)

    def step(self, next_pos: Point) -> Tuple[Point, bool]:
        """
        Attempt to move the robot to next_pos.

        Args:
            next_pos: Target grid coordinate for this step.

        Returns:
            Tuple of (new_position, collision_happened).
            If the move is blocked, returns (current_position, True).
        """
        if not self.grid.is_free(next_pos):
            logger.warning(
                "Collision: attempted %s -> %s (blocked). Robot stays at %s.",
                self.pos,
                next_pos,
                self.pos,
            )
            return self.pos, True

        self.pos = next_pos
        return self.pos, False
