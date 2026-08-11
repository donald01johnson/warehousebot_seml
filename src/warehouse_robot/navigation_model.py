"""
ML-based navigation model — path scoring and multi-candidate planning.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

from typing import List

from .config import DEFAULT_NUM_CANDIDATES
from .entities import Point
from .grid import WarehouseGrid
from .logging_config import get_logger
from .planner import bfs_shortest_path, greedy_random_path

logger = get_logger("navigation_model")


def score_path(path: List[Point], grid: WarehouseGrid) -> float:
    """
    Score a candidate path based on length and obstacle proximity.

    Scoring formula:
        score = path_length + (0.3 * number_of_adjacent_obstacle_cells)

    Lower score is better.  A path that runs close to obstacles receives
    a penalty because it is riskier in a real warehouse environment.

    Args:
        path: Ordered list of Points representing the candidate path.
        grid: The warehouse grid (used to look up adjacent obstacles).

    Returns:
        Float score — lower means safer and shorter.
    """
    if not path:
        logger.warning("score_path: empty path received. Returning inf.")
        return float("inf")

    base_score = float(len(path))
    obstacle_penalty = 0.0

    for point in path:
        r, c = point
        adjacent = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for ar, ac in adjacent:
            if (
                0 <= ar < grid.rows
                and 0 <= ac < grid.cols
                and grid.grid[ar, ac] == 1
            ):
                obstacle_penalty += 0.3

    score = base_score + obstacle_penalty

    logger.info(
        "score_path: path_len=%d | obstacle_penalty=%.2f | total=%.2f",
        len(path),
        obstacle_penalty,
        score,
    )
    return score


def plan_path_with_model(
    grid: WarehouseGrid,
    start: Point,
    goal: Point,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
) -> List[Point]:
    """
    Plan the best path by scoring multiple candidate paths.

    Strategy:
        1. Generate BFS optimal path as a guaranteed baseline.
        2. Generate num_candidates greedy-random paths for diversity.
        3. Score all paths that successfully reach the goal.
        4. Return the lowest-scoring (safest + shortest) path.

    Args:
        grid          : The warehouse grid environment.
        start         : Starting grid coordinate.
        goal          : Target grid coordinate.
        num_candidates: Number of greedy-random candidates to generate.

    Returns:
        Best-scoring path from start to goal.
        Falls back to BFS path if no candidates reach the goal.

    Raises:
        ValueError: If start or goal are invalid grid positions.
    """
    if not grid.is_free(start):
        logger.error(
            "plan_path_with_model: start %s is not free.", start)
        raise ValueError(f"Start position {start} is not a free cell.")

    if not grid.is_free(goal):
        logger.error(
            "plan_path_with_model: goal %s is not free.", goal)
        raise ValueError(f"Goal position {goal} is not a free cell.")

    logger.info(
        "plan_path_with_model: %s -> %s (candidates=%d)",
        start,
        goal,
        num_candidates,
    )

    # Always include BFS path as a reliable baseline
    bfs_path = bfs_shortest_path(grid, start, goal)
    candidates = [bfs_path] if bfs_path else []

    # Generate diverse greedy-random candidate paths
    for _ in range(num_candidates):
        path = greedy_random_path(grid, start, goal)
        if path and path[-1] == goal:
            candidates.append(path)

    if not candidates:
        logger.warning(
            "plan_path_with_model: no candidates found."
            "Returning empty path."
        )
        return []

    # Score only paths that reach the goal
    scored = [
        (score_path(p, grid), p)
        for p in candidates
        if p and p[-1] == goal]

    if not scored:
        logger.warning(
            "plan_path_with_model: no candidate reached goal."
            "Falling back to BFS."
        )
        return bfs_path

    scored.sort(key=lambda x: x[0])
    best_score, best_path = scored[0]

    logger.info(
        "plan_path_with_model: best path selected | score=%.2f | len=%d",
        best_score,
        len(best_path),
    )
    return best_path
