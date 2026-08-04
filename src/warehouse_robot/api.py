"""
FastAPI REST API for the Warehouse Robot Navigation System.

Endpoints:
    GET  /health        — service health check
    POST /predict-path  — plan optimal path from start to goal
    POST /score-path    — score a user-supplied path
    POST /simulate      — run a full navigation episode and return metrics

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 101
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .grid import WarehouseGrid
from .inference import execute_path
from .logging_config import get_logger, setup_logging
from .navigation_model import plan_path_with_model, score_path

setup_logging()
logger = get_logger("api")

app = FastAPI(
    title="Warehouse Robot Navigation API",
    description=(
        "REST API for autonomous warehouse robot path planning and"
        "simulation. "
        "Assignment II — AIMLCZG546, BITS Pilani WILP, Group 101."
    ),
    version="2.0.0",
)


# ─── Request / Response models ───

class PathRequest(BaseModel):
    rows: int = Field(default=10, ge=2, le=50, description="Grid rows")
    cols: int = Field(default=10, ge=2, le=50, description="Grid columns")
    obstacle_probability: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Obstacle probability (0–1)"
    )
    start: List[int] = Field(
        ..., min_length=2, max_length=2, description="[row, col]")
    goal: List[int] = Field(
        ..., min_length=2, max_length=2, description="[row, col]")
    num_candidates: int = Field(
        default=5, ge=1, le=20, description="Greedy-random candidates to score"
    )


class ScoreRequest(BaseModel):
    rows: int = Field(default=10, ge=2, le=50)
    cols: int = Field(default=10, ge=2, le=50)
    obstacle_probability: float = Field(default=0.2, ge=0.0, lt=1.0)
    path: List[List[int]] = Field(..., description="List of [row, col] steps")


class PathResponse(BaseModel):
    status: str
    path: Optional[List[List[int]]]
    path_length: int
    score: Optional[float]
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


# ─── Endpoints ───


@app.get("/health", tags=["Health"])
def health_check():
    """
    Verify that the API service is running.

    Returns:
        200 OK with service status and name.
    """
    logger.info("GET /health called.")
    return {"status": "healthy", "service": "warehouse-robot-navigation-api"}


@app.post("/predict-path", response_model=PathResponse, tags=["Planning"])
def predict_path(request: PathRequest):
    """
    Plan the optimal path from start to goal using ML-based candidate scoring.

    Generates several candidate paths (BFS + greedy-random), scores each one
    for length and obstacle proximity, and returns the best.

    Returns:
        200 — path found successfully.
        400 — start or goal position is invalid.
        404 — no feasible path exists.
        500 — internal server error.
    """
    logger.info(
        "POST /predict-path: start=%s, goal=%s", request.start,
        request.goal)

    try:
        start = tuple(request.start)
        goal = tuple(request.goal)

        grid = WarehouseGrid(
            rows=request.rows,
            cols=request.cols,
            obstacle_prob=request.obstacle_probability,
        )

        if not grid.is_free(start):
            logger.warning("/predict-path: start %s is not free.", start)
            raise HTTPException(
                status_code=400,
                detail=f"Start position {list(start)} is an obstacle or"
                "out of bounds.",
            )

        if not grid.is_free(goal):
            logger.warning("/predict-path: goal %s is not free.", goal)
            raise HTTPException(
                status_code=400,
                detail=f"Goal position {list(goal)} is an obstacle or"
                "out of bounds.",
            )

        path = plan_path_with_model(
            grid, start, goal, num_candidates=request.num_candidates
        )

        if not path:
            logger.warning(
                "/predict-path: no path found from %s to %s.", start, goal)
            raise HTTPException(
                status_code=404,
                detail=f"No feasible path found from {list(start)} to"
                "{list(goal)}.",
            )

        path_score = score_path(path, grid)

        return PathResponse(
            status="success",
            path=[list(p) for p in path],
            path_length=len(path),
            score=round(path_score, 4),
            message="Path planned successfully.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "/predict-path internal error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}")


@app.post("/score-path", tags=["Planning"])
def score_given_path(request: ScoreRequest):
    """
    Score a user-provided path on a freshly generated warehouse grid.

    Returns:
        200 — score computed successfully.
        500 — internal server error.
    """
    logger.info("POST /score-path: path_length=%d", len(request.path))

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
        }

    except Exception as exc:
        logger.error(
            "/score-path internal error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}")


@app.post("/simulate", response_model=SimulateResponse, tags=["Simulation"])
def simulate_task(request: PathRequest):
    """
    Run a complete navigation episode using BFS and robot simulation.

    Returns full episode metrics including travel time, collisions,
    and obstacle density.

    Returns:
        200 — simulation completed successfully.
        400 — start or goal position is invalid.
        500 — internal server error.
    """
    logger.info(
        "POST /simulate: start=%s, goal=%s", request.start, request.goal)

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
                detail=f"Start position {list(start)} is an obstacle or"
                "out of bounds.",
            )

        if not grid.is_free(goal):
            raise HTTPException(
                status_code=400,
                detail=f"Goal position {list(goal)} is an obstacle or"
                "out of bounds.",
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
            "/simulate internal error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}")
