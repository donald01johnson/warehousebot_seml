"""
Warehouse grid environment for robot navigation.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

import random
from typing import List

import numpy as np

from .entities import Point
from .logging_config import get_logger

logger = get_logger("grid")


class WarehouseGrid:
    """
    Represents a 2D warehouse grid environment.

    Convention:
        0 = free cell
        1 = obstacle cell

    Border cells are always kept free to guarantee connectivity.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacle_prob: float = 0.1,
    ):
        """
        Initialise a warehouse grid with random obstacle placement.

        Args:
            rows: Number of rows in the grid (must be >= 2).
            cols: Number of columns in the grid (must be >= 2).
            obstacle_prob: Probability that any interior cell is an obstacle.

        Raises:
            ValueError: If rows/cols are too small or obstacle_prob is invalid.
        """
        if rows < 2 or cols < 2:
            logger.error(
                "Grid dimensions too small: rows=%d, cols=%d", rows, cols)
            raise ValueError(
                f"Grid must be at least 2x2. Got rows={rows}, cols={cols}."
            )

        if not (0.0 <= obstacle_prob < 1.0):
            logger.error("Invalid obstacle probability: %.2f", obstacle_prob)
            raise ValueError(
                f"obstacle_prob must be in [0.0, 1.0). Got {obstacle_prob}."
            )

        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)

        # Place obstacles only in interior cells
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if random.random() < obstacle_prob:
                    self.grid[r, c] = 1

        logger.info(
            "WarehouseGrid initialised: %dx%d, obstacle_density=%.3f",
            rows,
            cols,
            self.obstacle_density(),
        )

    def is_free(self, p: Point) -> bool:
        """
        Check whether a cell is within bounds and not an obstacle.

        Args:
            p: Grid coordinate (row, col).

        Returns:
            True if the cell is free, False otherwise.
        """
        r, c = p
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return bool(self.grid[r, c] == 0)

    def obstacle_density(self) -> float:
        """Return the fraction of grid cells occupied by obstacles."""
        return float(self.grid.sum()) / (self.rows * self.cols)

    def display(
        self,
        robot: Point = None,
        start: Point = None,
        goal: Point = None,
    ):
        """
        Print an ASCII representation of the grid to stdout.

        Symbols:
            R = robot position
            S = start position
            G = goal position
            X = obstacle
            . = free cell
        """
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                p = (r, c)
                if p == robot:
                    row_str += "R "
                elif p == start:
                    row_str += "S "
                elif p == goal:
                    row_str += "G "
                elif self.grid[r, c] == 1:
                    row_str += "X "
                else:
                    row_str += ". "
            print(row_str)
        print()


def neighbors(grid: WarehouseGrid, p: Point) -> List[Point]:
    """
    Return all free adjacent cells (4-connected) for point p.

    Args:
        grid: The warehouse grid.
        p: Current grid coordinate.

    Returns:
        List of free neighbouring points.
    """
    r, c = p
    candidates = [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]
    return [n for n in candidates if grid.is_free(n)]
