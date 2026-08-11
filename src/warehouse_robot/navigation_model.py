"""
ML-based navigation model — path scoring and multi-candidate planning.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 212
"""

from typing import List

import numpy as np

from .config import DEFAULT_NUM_CANDIDATES, TIME_PER_STEP
from .entities import Point
from .grid import WarehouseGrid
from .logging_config import get_logger
from .planner import bfs_shortest_path, greedy_random_path

logger = get_logger("navigation_model")

# Feature order must match training.py FEATURE_COLUMNS exactly
_FEATURE_COLUMNS = [
    "path_len",
    "shortest_path_len",
    "obstacle_density",
    "collisions",
    "replans",
    "path_optimality_ratio",
    "time_per_step",
]


def score_path(path: List[Point], grid: WarehouseGrid) -> float:
    """
    Score a candidate path using a heuristic (fallback / baseline).

    Used when no trained model is available.
    Scoring formula:
        score = path_length + (0.3 * number_of_adjacent_obstacle_cells)

    Lower score is better.

    Args:
        path: Ordered list of Points representing the candidate path.
        grid: The warehouse grid.

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
        "score_path (heuristic): path_len=%d | "
        "obstacle_penalty=%.2f | total=%.2f",
        len(path),
        obstacle_penalty,
        score,
    )
    return score


def score_path_with_model(
    path: List[Point],
    grid: WarehouseGrid,
    model,
    shortest_path_len: int,
) -> float:
    """
    Score a candidate path using the trained ML model (RandomForest).

    Constructs the same feature vector used during training and calls
    model.predict() to estimate travel time for the given path.
    Falls back to the heuristic score_path() if model is None or
    prediction fails.

    This directly addresses the Assignment I feedback that scoring was
    a hardcoded heuristic — the trained RandomForestRegressor is now
    used for actual path scoring during inference.

    Args:
        path             : Ordered list of Points for the candidate path.
        grid             : The warehouse grid environment.
        model            : Fitted scikit-learn model (RandomForestRegressor).
                           If None, falls back to heuristic scoring.
        shortest_path_len: BFS optimal path length for optimality ratio.

    Returns:
        Predicted travel time (float) from the ML model.
        Falls back to heuristic score if model is unavailable.
    """
    if model is None or not path:
        logger.warning(
            "score_path_with_model: model is None or empty path. "
            "Falling back to heuristic."
        )
        return score_path(path, grid)

    try:
        path_len = len(path)
        optimality = (
            path_len / shortest_path_len
            if shortest_path_len > 0
            else 1.0
        )

        # Feature vector — must match training.py FEATURE_COLUMNS order
        features = np.array([[
            float(path_len),
            float(shortest_path_len),
            float(grid.obstacle_density()),
            0.0,                    # collisions unknown pre-execution
            0.0,                    # replans unknown pre-execution
            float(optimality),
            float(TIME_PER_STEP),   # time_per_step from config
        ]])

        prediction = float(model.predict(features)[0])

        logger.info(
            "score_path_with_model (ML): path_len=%d | "
            "predicted_travel_time=%.4f",
            path_len,
            prediction,
        )
        return prediction

    except Exception as exc:
        logger.warning(
            "score_path_with_model: ML prediction failed (%s). "
            "Falling back to heuristic.",
            str(exc),
        )
        return score_path(path, grid)


def plan_path_with_model(
    grid: WarehouseGrid,
    start: Point,
    goal: Point,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    model=None,
) -> List[Point]:
    """
    Plan the best path by scoring multiple candidate paths.

    If a trained model is provided, candidate paths are scored using
    model.predict() on engineered features (ML-based scoring).
    If no model is provided, falls back to the heuristic score_path().

    Strategy:
        1. Generate BFS optimal path as a guaranteed baseline.
        2. Generate num_candidates greedy-random paths for diversity.
        3. Score all paths that reach the goal using ML or heuristic.
        4. Return the lowest-scoring (best predicted travel time) path.

    Args:
        grid          : The warehouse grid environment.
        start         : Starting grid coordinate.
        goal          : Target grid coordinate.
        num_candidates: Number of greedy-random candidates to generate.
        model         : Fitted RandomForestRegressor for ML scoring.
                        If None, uses heuristic scoring as fallback.

    Returns:
        Best-scoring path from start to goal.
        Falls back to BFS path if no candidates reach the goal.

    Raises:
        ValueError: If start or goal are invalid grid positions.
    """
    if not grid.is_free(start):
        logger.error(
            "plan_path_with_model: start %s is not free.", start
        )
        raise ValueError(f"Start position {start} is not a free cell.")

    if not grid.is_free(goal):
        logger.error(
            "plan_path_with_model: goal %s is not free.", goal
        )
        raise ValueError(f"Goal position {goal} is not a free cell.")

    scoring_mode = "ML model" if model is not None else "heuristic"
    logger.info(
        "plan_path_with_model: %s -> %s (candidates=%d, scoring=%s)",
        start,
        goal,
        num_candidates,
        scoring_mode,
    )

    # Always include BFS path as a reliable baseline
    bfs_path = bfs_shortest_path(grid, start, goal)
    candidates = [bfs_path] if bfs_path else []
    shortest_len = len(bfs_path) if bfs_path else 0

    # Generate diverse greedy-random candidate paths
    for _ in range(num_candidates):
        path = greedy_random_path(grid, start, goal)
        if path and path[-1] == goal:
            candidates.append(path)

    if not candidates:
        logger.warning(
            "plan_path_with_model: no candidates found. "
            "Returning empty path."
        )
        return []

    # Score candidates using ML model if available, else heuristic
    scored = [
        (
            score_path_with_model(p, grid, model, shortest_len)
            if model is not None
            else score_path(p, grid),
            p,
        )
        for p in candidates
        if p and p[-1] == goal
    ]

    if not scored:
        logger.warning(
            "plan_path_with_model: no candidate reached goal. "
            "Falling back to BFS."
        )
        return bfs_path

    scored.sort(key=lambda x: x[0])
    best_score, best_path = scored[0]

    logger.info(
        "plan_path_with_model: best path | "
        "score=%.4f | len=%d | scoring=%s",
        best_score,
        len(best_path),
        scoring_mode,
    )
    return best_path
