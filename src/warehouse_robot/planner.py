"""
Path planning algorithms for warehouse robot navigation.

Research Code vs Production Code — Assignment II Requirement 2
--------------------------------------------------------------
RESEARCH (notebook) version characteristics:
  - Uses bare global MOVES constant directly in function body
  - No input validation whatsoever
  - Print-based diagnostics, no structured logging
  - No docstrings or type hints
  - Runs inline inside a notebook cell

PRODUCTION (this module) version characteristics:
  - Full input validation with descriptive ValueError messages
  - INFO / WARNING / ERROR structured logging via logging_config
  - Complete type hints on every function signature
  - Comprehensive docstrings
  - Importable, testable, reusable module

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 101
"""

import random
from collections import deque
from typing import List

from .config import MAX_GREEDY_STEPS, MOVES
from .entities import Point
from .grid import WarehouseGrid
from .logging_config import get_logger

logger = get_logger("planner")


def bfs_shortest_path(
    grid: WarehouseGrid,
    start: Point,
    goal: Point,
) -> List[Point]:
    """
    Find the shortest path from start to goal using Breadth-First Search.

    This is the PRODUCTION version of the Assignment I notebook function.
    It adds input validation and structured logging on top of the original
    BFS logic, making it safe to call from tests and the REST API.

    Args:
        grid : The warehouse grid environment.
        start: Starting grid coordinate (row, col).
        goal : Target grid coordinate (row, col).

    Returns:
        Ordered list of Points from start to goal (inclusive).
        Returns an empty list if no path exists.

    Raises:
        ValueError: If start or goal are blocked or out of bounds.
    """
    if not grid.is_free(start):
        logger.error("BFS start position is not free: %s", start)
        raise ValueError(f"Start position {start} is not a free cell.")

    if not grid.is_free(goal):
        logger.error("BFS goal position is not free: %s", goal)
        raise ValueError(f"Goal position {goal} is not a free cell.")

    if start == goal:
        logger.info(
            "BFS: start equals goal %s. Returning trivial path.", start)
        return [start]

    logger.info("BFS planning path: %s -> %s", start, goal)

    queue = deque([start])
    visited = {start: None}  # maps node -> parent node

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)
            if grid.is_free(nxt) and nxt not in visited:
                visited[nxt] = current
                queue.append(nxt)

    if goal not in visited:
        logger.warning(
            "BFS: no path found from %s to %s (goal unreachable).", start, goal
        )
        return []

    # Reconstruct path by tracing parent pointers
    path: List[Point] = []
    p = goal
    while p is not None:
        path.append(p)
        p = visited[p]
    path.reverse()

    logger.info(
        "BFS path found: length=%d, start=%s, goal=%s",
        len(path),
        start,
        goal,
    )
    return path


def greedy_random_path(
    grid: WarehouseGrid,
    start: Point,
    goal: Point,
    max_steps: int = MAX_GREEDY_STEPS,
) -> List[Point]:
    """
    Generate a candidate path using a greedy random walk towards the goal.

    Used by plan_path_with_model() to create diverse candidate paths for
    ML-based scoring. Unlike BFS, this may not always reach the goal,
    but generates varied paths for model input.

    Args:
        grid     : The warehouse grid environment.
        start    : Starting grid coordinate.
        goal     : Target grid coordinate.
        max_steps: Maximum steps before giving up.

    Returns:
        List of Points starting from start.
        The path reaches goal only if the walk converges.

    Raises:
        ValueError: If start is a blocked or out-of-bounds cell.
    """
    if not grid.is_free(start):
        logger.error(
            "greedy_random_path: start %s is not a free cell.", start)
        raise ValueError(f"Start position {start} is not a free cell.")

    logger.info(
        "Greedy random path: %s -> %s (max_steps=%d)", start, goal, max_steps)

    path = [start]
    current = start
    visited = {start}
    gr, gc = goal

    for _ in range(max_steps):
        if current == goal:
            break

        r, c = current
        candidates = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]

        free_candidates = [
            n for n in candidates if grid.is_free(n) and n not in visited
        ]

        if not free_candidates:
            logger.warning(
                "Greedy path stuck at %s — no unvisited free neighbours.",
                current
            )
            break

        # Sort by Manhattan distance to goal + small random jitter
        def _heuristic(p):
            return abs(p[0] - gr) + abs(p[1] - gc) + random.uniform(0, 1.5)

        free_candidates.sort(key=_heuristic)

        current = free_candidates[0]
        path.append(current)
        visited.add(current)

    if current == goal:
        logger.info("Greedy path reached goal: length=%d", len(path))
    else:
        logger.warning(
            "Greedy path did NOT reach goal. length=%d, last=%s, goal=%s",
            len(path),
            current,
            goal,
        )

    return path
