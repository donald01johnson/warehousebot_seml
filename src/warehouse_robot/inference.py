"""
Path inference and episode execution for warehouse robot navigation.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

import random
from typing import List

from .config import TIME_PER_STEP
from .entities import EpisodeLog, Point
from .grid import WarehouseGrid
from .logging_config import get_logger
from .planner import bfs_shortest_path
from .simulator import RobotSimulator

logger = get_logger("inference")

# In-memory episode store (mirrors Assignment I RAW_LOGS global)
RAW_LOGS: List[EpisodeLog] = []


def execute_path(
    grid: WarehouseGrid,
    start: Point,
    goal: Point,
) -> EpisodeLog:
    """
    Execute a navigation task from start to goal and return an episode log.

    Uses BFS to compute the optimal path, then simulates step-by-step
    robot traversal, recording collisions and travel time.

    Args:
        grid : The warehouse grid environment.
        start: Starting grid coordinate.
        goal : Target grid coordinate.

    Returns:
        EpisodeLog capturing the full outcome of the navigation episode.

    Raises:
        ValueError: If start or goal are invalid.
        RuntimeError: On unexpected simulation failure.
    """
    logger.info("execute_path called: %s -> %s", start, goal)

    try:
        robot = RobotSimulator(grid, start)
        path = bfs_shortest_path(grid, start, goal)
        shortest_len = len(path) if path else 0

        travel_time = 0.0
        collisions = 0
        replans = 0  # reserved for future replanning logic

        if not path:
            logger.warning(
                "No valid path from %s to %s. Logging failure episode.",
                start,
                goal,
            )
            return EpisodeLog(
                start=start,
                goal=goal,
                path=[],
                shortest_path_len=0,
                travel_time=0.0,
                collisions=0,
                replans=0,
                obstacle_density=grid.obstacle_density(),
            )

        for next_pos in path[1:]:
            _, collided = robot.step(next_pos)
            travel_time += TIME_PER_STEP
            if collided:
                collisions += 1

        logger.info(
            "execute_path complete: "
            "travel_time=%.1f, collisions=%d, path_len=%d",
            travel_time,
            collisions,
            len(path),
        )

        return EpisodeLog(
            start=start,
            goal=goal,
            path=path,
            shortest_path_len=shortest_len,
            travel_time=travel_time,
            collisions=collisions,
            replans=replans,
            obstacle_density=grid.obstacle_density(),
        )

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error in execute_path: %s", str(exc), exc_info=True)
        raise RuntimeError(f"execute_path failed: {exc}") from exc


def run_random_task_and_log(grid: WarehouseGrid) -> EpisodeLog:
    """
    Pick a random valid (start, goal) pair,
    execute navigation, and log the episode.

    Args:
        grid: The warehouse grid environment.

    Returns:
        EpisodeLog for the completed episode.

    Raises:
        RuntimeError: If no valid (start, goal) pair
        is found within the attempt limit.
    """
    max_attempts = 1000

    for attempt in range(max_attempts):
        start = (
            random.randint(0, grid.rows - 1),
            random.randint(0, grid.cols - 1),
        )
        goal = (
            random.randint(0, grid.rows - 1),
            random.randint(0, grid.cols - 1),
        )
        if grid.is_free(start) and grid.is_free(goal) and start != goal:
            break
    else:
        logger.error(
            "Could not find valid start/goal after %d attempts.", max_attempts)
        raise RuntimeError(
            f"Failed to find a valid (start, goal) pair "
            f"after {max_attempts} attempts."
        )

    logger.info("Random task selected: start=%s, goal=%s", start, goal)

    ep = execute_path(grid, start, goal)
    RAW_LOGS.append(ep)
    return ep
