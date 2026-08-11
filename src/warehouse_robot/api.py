"""
FastAPI REST API for the Warehouse Robot Navigation System.

Endpoints:
    GET  /health        -- service health check
    POST /predict-path  -- plan optimal path using trained ML model
    POST /score-path    -- score a user-supplied path
    POST /simulate      -- run a full navigation episode

The API trains a RandomForestRegressor at startup using simulated
episodes. The /predict-path endpoint scores candidate paths using
model.predict() on engineered features rather than a hardcoded
heuristic -- directly addressing Assignment I feedback.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_COLS,
    DEFAULT_NUM_EPISODES,
    DEFAULT_OBSTACLE_PROB,
    DEFAULT_ROWS,
)
from .data_ingestion import clean_episodes, collect_logs
from .feature_engineering import compute_labels, engineer_features
from .grid import WarehouseGrid
from .inference import RAW_LOGS, execute_path, run_random_task_and_log
from .logging_config import get_logger, setup_logging
from .navigation_model import plan_path_with_model, score_path
from .training import split_dataset, train_model

setup_logging()
logger = get_logger("api")


# ── Startup: Train model ───────────────────────────────────────────


def _train_startup_model():
    """
    Train a RandomForestRegressor at API startup.

    Simulates DEFAULT_NUM_EPISODES navigation episodes, runs the full
    data pipeline (F1-F5), trains a RandomForest regression model,
    and returns the fitted model.

    Returns:
        Fitted RandomForestRegressor, or None if training fails.
    """
    try:
        logger.info(
            "API startup: training RandomForest model "
            "on %d episodes...",
            DEFAULT_NUM_EPISODES,
        )
        RAW_LOGS.clear()
        grid = WarehouseGrid(
            rows=DEFAULT_ROWS,
            cols=DEFAULT_COLS,
            obstacle_prob=DEFAULT_OBSTACLE_PROB,
        )
        for _ in range(DEFAULT_NUM_EPISODES):
            run_random_task_and_log(grid)

        episodes = collect_logs(RAW_LOGS)
        episodes = clean_episodes(episodes)
        episodes = engineer_features(episodes)
        episodes = compute_labels(episodes)
        train, _, _ = split_dataset(episodes)

        model = train_model(train, model_type="random_forest")
        logger.info(
            "API startup: RandomForest model trained successfully "
            "on %d episodes.",
            len(train),
        )
        return model

    except Exception as exc:
        logger.error(
            "API startup: model training failed: %s. "
            "API will use heuristic fallback.",
            str(exc),
            exc_info=True,
        )
        return None


# Train once at import time — available to all requests
_startup_model = _train_startup_model()


# ── FastAPI App ────────────────────────────────────────────────────


app = FastAPI(
    title="Warehouse Robot Navigation API",
    description=(
        "REST API for autonomous warehouse robot path planning "
        "and simulation. Uses a trained RandomForest model for "
        "ML-based path scoring. "
        "Assignment II — AIMLCZG546, BITS Pilani WILP, Group 212."
    ),
    version="2.0.0",
)


# ── Request / Response models ──────────────────────────────────────


class PathRequest(BaseModel):
    rows: int = Field(
        default=10, ge=2, le=50, description="Grid rows"
    )
    cols: int = Field(
        default=10, ge=2, le=50, description="Grid columns"
    )
    obstacle_probability: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Obstacle probability (0-1)",
    )
    start: List[int] = Field(
        ..., min_length=2, max_length=2, description="[row, col]"
    )
    goal: List[int] = Field(
        ..., min_length=2, max_length=2, description="[row, col]"
    )
    num_candidates: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Greedy-random candidates to score",
    )


class ScoreRequest(BaseModel):
    rows: int = Field(default=10, ge=2, le=50)
    cols: int = Field(default=10, ge=2, le=50)
    obstacle_probability: float = Field(
        default=0.2, ge=0.0, lt=1.0
    )
    path: List[List[int]] = Field(
        ..., description="List of [row, col] steps"
    )


class PathResponse(BaseModel):
    status: str
    path: Optional[List[List[int]]]
    path_length: int
    score: Optional[float]
    scoring_method: str
    message: str


class SimulateResponse(BaseModel):
    status: str
    start: List[int]
    goal: List[int]
    path: List[List[int]]
    path_length: int
    travel_time: float
    collisions: int
    obstacle_density: float
    message: str


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
def health_check():
    """
    Verify that the API service is running.

    Returns:
        200 OK with service status, name, and model availability.
    """
    logger.info("GET /health called.")
    return {
        "status": "healthy",
        "service": "warehouse-robot-navigation-api",
        "model_available": _startup_model is not None,
        "scoring_method": (
            "RandomForest ML model"
            if _startup_model is not None
            else "heuristic fallback"
        ),
    }


@app.post(
    "/predict-path", response_model=PathResponse, tags=["Planning"]
)
def predict_path(request: PathRequest):
    """
    Plan the optimal path using trained ML model scoring.

    Generates BFS + greedy-random candidate paths, scores each using
    the trained RandomForestRegressor (model.predict() on engineered
    features), and returns the best-scoring path.

    Falls back to heuristic scoring if startup model is unavailable.

    Returns:
        200 -- path found successfully.
        400 -- start or goal position is invalid.
        404 -- no feasible path exists.
        500 -- internal server error.
    """
    logger.info(
        "POST /predict-path: start=%s, goal=%s",
        request.start,
        request.goal,
    )

    try:
        start = tuple(request.start)
        goal = tuple(request.goal)

        grid = WarehouseGrid(
            rows=request.rows,
            cols=request.cols,
            obstacle_prob=request.obstacle_probability,
        )

        if not grid.is_free(start):
            logger.warning(
                "/predict-path: start %s is not free.", start
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Start position {list(start)} is an obstacle "
                    "or out of bounds."
                ),
            )

        if not grid.is_free(goal):
            logger.warning(
                "/predict-path: goal %s is not free.", goal
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Goal position {list(goal)} is an obstacle "
                    "or out of bounds."
                ),
            )

        # Use trained ML model for scoring (heuristic if unavailable)
        path = plan_path_with_model(
            grid,
            start,
            goal,
            num_candidates=request.num_candidates,
            model=_startup_model,
        )

        if not path:
            logger.warning(
                "/predict-path: no path found from %s to %s.",
                start,
                goal,
            )
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No feasible path found from "
                    f"{list(start)} to {list(goal)}."
                ),
            )

        # Compute final score for response
        from .navigation_model import score_path_with_model
        from .planner import bfs_shortest_path as _bfs

        bfs_ref = _bfs(grid, start, goal)
        path_score = score_path_with_model(
            path, grid, _startup_model, len(bfs_ref)
        )

        scoring_method = (
            "RandomForest ML model"
            if _startup_model is not None
            else "heuristic fallback"
        )

        return PathResponse(
            status="success",
            path=[list(p) for p in path],
            path_length=len(path),
            score=round(path_score, 4),
            scoring_method=scoring_method,
            message="Path planned successfully.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "/predict-path internal error: %s",
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        )


@app.post("/score-path", tags=["Planning"])
def score_given_path(request: ScoreRequest):
    """
    Score a user-provided path on a freshly generated warehouse grid.

    Returns:
        200 -- score computed successfully.
        500 -- internal server error.
    """
    logger.info(
        "POST /score-path: path_length=%d", len(request.path)
    )

    try:
        grid = WarehouseGrid(
            rows=request.rows,
            cols=request.cols,
            obstacle_prob=request.obstacle_probability,
        )

        path = [tuple(p) for p in request.path]
        path_score = score_path(path, grid)

        return {
            "status": "success",
            "path_length": len(path),
            "score": round(path_score, 4),
            "scoring_method": "heuristic",
        }

    except Exception as exc:
        logger.error(
            "/score-path internal error: %s",
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        )


@app.post(
    "/simulate", response_model=SimulateResponse, tags=["Simulation"]
)
def simulate_task(request: PathRequest):
    """
    Run a complete navigation episode using BFS and robot simulation.

    Returns full episode metrics including travel time, collisions,
    and obstacle density.

    Returns:
        200 -- simulation completed successfully.
        400 -- start or goal position is invalid.
        500 -- internal server error.
    """
    logger.info(
        "POST /simulate: start=%s, goal=%s",
        request.start,
        request.goal,
    )

    try:
        start = tuple(request.start)
        goal = tuple(request.goal)

        grid = WarehouseGrid(
            rows=request.rows,
            cols=request.cols,
            obstacle_prob=request.obstacle_probability,
        )

        if not grid.is_free(start):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Start position {list(start)} is an obstacle "
                    "or out of bounds."
                ),
            )

        if not grid.is_free(goal):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Goal position {list(goal)} is an obstacle "
                    "or out of bounds."
                ),
            )

        ep = execute_path(grid, start, goal)

        return SimulateResponse(
            status="success",
            start=list(ep.start),
            goal=list(ep.goal),
            path=[list(p) for p in ep.path],
            path_length=len(ep.path),
            travel_time=ep.travel_time,
            collisions=ep.collisions,
            obstacle_density=round(ep.obstacle_density, 4),
            message="Simulation complete.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "/simulate internal error: %s",
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        )
