import random
import numpy as np

from .entities import Point


class WarehouseGrid:
    def __init__(
        self,
        rows: int,
        cols: int,
        obstacle_prob: float = 0.1
    ):
        self.rows = rows
        self.cols = cols

        self.grid = np.zeros(
            (rows, cols),
            dtype=int
        )

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if random.random() < obstacle_prob:
                    self.grid[r, c] = 1

    def is_free(self, p: Point) -> bool:
        r, c = p

        if (
            r < 0
            or r >= self.rows
            or c < 0
            or c >= self.cols
        ):
            return False

        return self.grid[r, c] == 0

    def obstacle_density(self) -> float:
        return float(
            self.grid.sum()
        ) / (
            self.rows * self.cols
        )

    def display(
        self,
        robot: Point = None,
        start: Point = None,
        goal: Point = None,
    ):
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


def neighbors(
    grid: WarehouseGrid,
    p: Point,
):
    r, c = p

    candidates = [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]

    return [
        n
        for n in candidates
        if grid.is_free(n)
    ]
