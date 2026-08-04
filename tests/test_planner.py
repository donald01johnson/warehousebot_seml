"""
Unit tests for BFS path planner and greedy random path.

Assignment II Requirement 6 - Unit Tests
Assignment II - AIMLCZG546, BITS Pilani WILP | Group 101
"""

import pytest

from src.warehouse_robot.grid import WarehouseGrid
from src.warehouse_robot.planner import (
    bfs_shortest_path,
    greedy_random_path,
)


def open_grid(rows=8, cols=8):
    return WarehouseGrid(rows=rows, cols=cols, obstacle_prob=0.0)


# --- BFS Shortest Path ---


class TestBFSShortestPath:

    def test_returns_correct_start(self):
        grid = open_grid()
        path = bfs_shortest_path(grid, (0, 0), (0, 4))
        assert path[0] == (0, 0)

    def test_returns_correct_goal(self):
        grid = open_grid()
        path = bfs_shortest_path(grid, (0, 0), (0, 4))
        assert path[-1] == (0, 4)

    def test_horizontal_path_length(self):
        grid = open_grid(rows=5, cols=5)
        path = bfs_shortest_path(grid, (0, 0), (0, 4))
        assert len(path) == 5

    def test_start_equals_goal_trivial_path(self):
        grid = open_grid()
        path = bfs_shortest_path(grid, (2, 2), (2, 2))
        assert path == [(2, 2)]

    def test_path_is_connected(self):
        """Each step must be adjacent (Manhattan distance == 1)."""
        grid = open_grid(rows=8, cols=8)
        path = bfs_shortest_path(grid, (0, 0), (7, 7))
        assert path is not None and len(path) > 0
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            dist = abs(r1 - r2) + abs(c1 - c2)
            assert dist == 1, (
                f"Non-adjacent steps: {path[i]} -> {path[i + 1]}"
            )

    def test_all_cells_are_free(self):
        grid = open_grid(rows=8, cols=8)
        path = bfs_shortest_path(grid, (0, 0), (7, 7))
        for p in path:
            assert grid.is_free(p), (
                f"Path passes through blocked cell {p}"
            )

    def test_no_path_when_blocked(self):
        grid = open_grid(rows=5, cols=5)
        for c in range(5):
            grid.grid[2][c] = 1
        path = bfs_shortest_path(grid, (0, 0), (4, 4))
        assert path == []

    def test_blocked_start_raises_value_error(self):
        grid = open_grid(rows=5, cols=5)
        grid.grid[0][0] = 1
        with pytest.raises(ValueError, match="Start position"):
            bfs_shortest_path(grid, (0, 0), (4, 4))

    def test_blocked_goal_raises_value_error(self):
        grid = open_grid(rows=5, cols=5)
        grid.grid[4][4] = 1
        with pytest.raises(ValueError, match="Goal position"):
            bfs_shortest_path(grid, (0, 0), (4, 4))

    def test_bfs_finds_shortest_path(self):
        """BFS must find the optimal shortest path."""
        grid = open_grid(rows=5, cols=5)
        path = bfs_shortest_path(grid, (0, 0), (4, 4))
        # Manhattan dist (0,0)->(4,4) = 8, optimal path = 9 cells
        assert len(path) == 9


# --- Greedy Random Path ---


class TestGreedyRandomPath:

    def test_starts_at_start(self):
        grid = open_grid()
        path = greedy_random_path(grid, (0, 0), (7, 7))
        assert path[0] == (0, 0)

    def test_path_is_non_empty(self):
        grid = open_grid()
        path = greedy_random_path(grid, (0, 0), (7, 7))
        assert len(path) > 0

    def test_all_cells_are_free(self):
        grid = open_grid()
        path = greedy_random_path(grid, (0, 0), (7, 7))
        for p in path:
            assert grid.is_free(p), (
                f"Greedy path hit blocked cell {p}"
            )

    def test_blocked_start_raises(self):
        grid = open_grid(rows=5, cols=5)
        grid.grid[0][0] = 1
        with pytest.raises(ValueError):
            greedy_random_path(grid, (0, 0), (4, 4))

    def test_path_reaches_goal_on_open_grid(self):
        """On a fully open grid the greedy walk reaches the goal."""
        grid = open_grid(rows=6, cols=6)
        path = greedy_random_path(grid, (0, 0), (5, 5))
        assert path[-1] == (5, 5), (
            f"Expected goal (5,5), got {path[-1]}"
        )
