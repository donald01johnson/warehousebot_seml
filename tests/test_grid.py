"""
Unit tests for WarehouseGrid and neighbors().

Assignment II Requirement 6 - Unit Tests
Assignment II - AIMLCZG546, BITS Pilani WILP | Group 212
"""

import pytest

from src.warehouse_robot.grid import WarehouseGrid, neighbors


# --- WarehouseGrid ---


class TestWarehouseGridInit:

    def test_correct_dimensions(self):
        grid = WarehouseGrid(rows=6, cols=8, obstacle_prob=0.0)
        assert grid.rows == 6
        assert grid.cols == 8

    def test_no_obstacles_when_prob_zero(self):
        grid = WarehouseGrid(rows=6, cols=6, obstacle_prob=0.0)
        assert grid.obstacle_density() == 0.0

    def test_all_interior_obstacles_when_prob_one_minus_epsilon(self):
        grid = WarehouseGrid(rows=6, cols=6, obstacle_prob=0.999)
        assert grid.obstacle_density() > 0.0

    def test_invalid_rows_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            WarehouseGrid(rows=1, cols=5)

    def test_invalid_cols_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            WarehouseGrid(rows=5, cols=1)

    def test_negative_obstacle_prob_raises(self):
        with pytest.raises(ValueError, match="obstacle_prob"):
            WarehouseGrid(rows=5, cols=5, obstacle_prob=-0.1)

    def test_obstacle_prob_one_raises(self):
        with pytest.raises(ValueError, match="obstacle_prob"):
            WarehouseGrid(rows=5, cols=5, obstacle_prob=1.0)

    def test_border_cells_always_free(self):
        """Border cells must never be obstacles regardless of prob."""
        grid = WarehouseGrid(rows=6, cols=6, obstacle_prob=0.999)
        for r in range(grid.rows):
            assert grid.is_free((r, 0))
            assert grid.is_free((r, grid.cols - 1))
        for c in range(grid.cols):
            assert grid.is_free((0, c))
            assert grid.is_free((grid.rows - 1, c))


class TestWarehouseGridIsFree:

    def test_free_corner_top_left(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        assert grid.is_free((0, 0)) == True  # noqa: E712

    def test_free_corner_bottom_right(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        assert grid.is_free((4, 4)) == True  # noqa: E712

    def test_out_of_bounds_negative(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        assert grid.is_free((-1, 0)) is False
        assert grid.is_free((0, -1)) is False

    def test_out_of_bounds_exceeds_size(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        assert grid.is_free((5, 0)) is False
        assert grid.is_free((0, 5)) is False

    def test_obstacle_cell_is_not_free(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        grid.grid[2][2] = 1
        assert grid.is_free((2, 2)) == False  # noqa: E712


class TestWarehouseGridDisplay:

    def test_display_runs_without_error(self, capsys):
        grid = WarehouseGrid(rows=4, cols=4, obstacle_prob=0.0)
        grid.display(robot=(0, 0), start=(0, 0), goal=(3, 3))
        captured = capsys.readouterr()
        assert "R" in captured.out
        assert "G" in captured.out

    def test_display_shows_obstacle(self, capsys):
        grid = WarehouseGrid(rows=4, cols=4, obstacle_prob=0.0)
        grid.grid[2][2] = 1
        grid.display()
        captured = capsys.readouterr()
        assert "X" in captured.out


# --- neighbors() ---


class TestNeighbors:

    def test_center_cell_has_four_neighbours(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        result = set(neighbors(grid, (2, 2)))
        assert result == {(1, 2), (3, 2), (2, 1), (2, 3)}

    def test_corner_cell_has_two_neighbours(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        result = set(neighbors(grid, (0, 0)))
        assert result == {(1, 0), (0, 1)}

    def test_blocked_neighbours_excluded(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        grid.grid[1][2] = 1
        grid.grid[3][2] = 1
        result = neighbors(grid, (2, 2))
        assert (1, 2) not in result
        assert (3, 2) not in result
        assert len(result) == 2

    def test_all_neighbours_blocked_returns_empty(self):
        grid = WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)
        grid.grid[1][2] = 1
        grid.grid[3][2] = 1
        grid.grid[2][1] = 1
        grid.grid[2][3] = 1
        result = neighbors(grid, (2, 2))
        assert result == []
