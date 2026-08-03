"""
Pytest configuration and shared fixtures for Assignment II test suite.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 101
"""

import random

import pytest

from src.warehouse_robot.data_ingestion import (
    clean_episodes,
    collect_logs,
)
from src.warehouse_robot.entities import EpisodeLog
from src.warehouse_robot.feature_engineering import (
    compute_labels,
    engineer_features,
)
from src.warehouse_robot.grid import WarehouseGrid


# ── Shared grid fixtures ───────────────────────────────────────────


@pytest.fixture
def open_grid():
    """8x8 grid with no obstacles — guarantees path exists."""
    return WarehouseGrid(rows=8, cols=8, obstacle_prob=0.0)


@pytest.fixture
def small_grid():
    """5x5 grid with no obstacles for lightweight tests."""
    return WarehouseGrid(rows=5, cols=5, obstacle_prob=0.0)


# ── Shared episode helpers ─────────────────────────────────────────


def make_episode(
    path_len: int = 5,
    travel_time: float = 5.0,
    collisions: int = 0,
    replans: int = 0,
    obstacle_density: float = 0.1,
) -> EpisodeLog:
    """Create a synthetic EpisodeLog for testing."""
    path = [(i, 0) for i in range(path_len)]
    return EpisodeLog(
        start=(0, 0),
        goal=(path_len - 1, 0),
        path=path,
        shortest_path_len=path_len,
        travel_time=travel_time,
        collisions=collisions,
        replans=replans,
        obstacle_density=obstacle_density,
    )


def make_labeled_episodes(n: int = 60):
    """
    Generate n fully-labeled episodes ready for ML training.
    Episodes have varied path lengths for realistic training data.
    """
    raw = []
    for _ in range(n):
        path_len = random.randint(3, 15)
        raw.append(
            make_episode(
                path_len=path_len,
                travel_time=float(path_len),
            )
        )
    collected = collect_logs(raw)
    cleaned = clean_episodes(collected)
    featured = engineer_features(cleaned)
    return compute_labels(featured)


@pytest.fixture
def labeled_episodes():
    """Fixture: 80 fully-labeled episodes for ML tests."""
    return make_labeled_episodes(80)
