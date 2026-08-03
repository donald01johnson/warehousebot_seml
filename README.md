# Warehouse Robot Navigation System — Assignment II

**Course**      : AIML ZG535 — Edge AI Systems  
**Institution** : BITS Pilani WILP  
**Group**       : 101  
**Members**     : Donald Johnson A, Sanchi Jain, Brijesh Pandey, Gochhayat Avimanyu  
**Deadline**    : 15 August 2026, 23:00

---

## Overview

Production-grade refactoring of the Assignment I autonomous warehouse robot
navigation system. Implements modular package design, REST API, comprehensive
test suite, ML training/evaluation, code quality tooling, and structured logging.

---

## Quick Start

```bash
# 1. Activate environment
source whousebot/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the full demo pipeline
python -m src.warehouse_robot.demo

# 4. Run all tests
pytest -v

# 5. Launch the REST API (Swagger UI at http://127.0.0.1:8000/docs)
uvicorn src.warehouse_robot.api:app --reload

# 6. Code quality checks
isort src tests
black src tests
flake8 src tests
```

---

## Project Structure

```
whousebot_seml/
├── Group_101.ipynb              # Main assignment notebook
├── requirements.txt
├── pyproject.toml
├── README.md
│
├── src/
│   └── warehouse_robot/         # Production package
│       ├── __init__.py
│       ├── config.py            # Global constants
│       ├── entities.py          # EpisodeLog dataclass + Point type
│       ├── grid.py              # WarehouseGrid + neighbors()
│       ├── simulator.py         # RobotSimulator
│       ├── planner.py           # bfs_shortest_path + greedy_random_path
│       ├── inference.py         # execute_path + run_random_task_and_log
│       ├── data_ingestion.py    # F1: collect_logs, F2: clean_episodes
│       ├── feature_engineering.py # F3: engineer_features, F4: compute_labels
│       ├── training.py          # F5: split_dataset + ML train/evaluate
│       ├── navigation_model.py  # score_path + plan_path_with_model
│       ├── logging_config.py    # Centralised logging setup
│       ├── api.py               # FastAPI REST endpoints
│       └── demo.py              # main_demo() end-to-end pipeline
│
├── tests/
│   ├── conftest.py
│   ├── test_grid.py             # Unit: WarehouseGrid, neighbors
│   ├── test_planner.py          # Unit: BFS, greedy path
│   ├── test_data_pipeline.py    # Unit + Integration: F1–F5
│   ├── test_training.py         # ML training + inference tests
│   └── test_api.py              # API integration tests
│
├── reports/                     # Assignment report artifacts
├── screenshots/                 # Evidence screenshots for report
├── artifacts/                   # Model outputs, logs
└── logs/                        # Runtime log files
```

---

## Assignment II Coverage

| Requirement | Implementation | File(s) |
|-------------|---------------|---------|
| Modular OOP design | Full production package | `src/warehouse_robot/` |
| Research vs Production | Notebook vs module | `Group_101.ipynb` vs `planner.py` |
| Error handling | All critical functions | All modules |
| Logging (INFO/WARN/ERROR) | Centralised + per-module | `logging_config.py` |
| Code formatting | black + isort + flake8 | `pyproject.toml` |
| REST API | FastAPI 3 endpoints | `api.py` |
| Unit tests | 40+ test cases | `test_grid.py`, `test_planner.py` |
| Integration tests | Full pipeline + API | `test_data_pipeline.py`, `test_api.py` |
| ML training tests | Overfit + invariance | `test_training.py` |
| ML inference tests | Direction + range | `test_training.py` |
| Model quality (MAE/RMSE/R2) | evaluate_model() | `training.py` |
| Data quality metrics | invalid_rate, missing_rate | `test_data_pipeline.py` |
| Production testing strategy | Shadow + canary + A/B | Report Section 9 |
| Security consideration | Input validation | `api.py`, `grid.py`, `planner.py` |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict-path` | ML-scored optimal path planning |
| POST | `/score-path` | Score a user-provided path |
| POST | `/simulate` | Full BFS + simulation episode |

Swagger UI available at: `http://127.0.0.1:8000/docs`

---

## Example API Request

```bash
curl -X POST http://127.0.0.1:8000/predict-path \
  -H "Content-Type: application/json" \
  -d '{
    "rows": 10,
    "cols": 10,
    "obstacle_probability": 0.2,
    "start": [0, 0],
    "goal": [9, 9],
    "num_candidates": 5
  }'
```

---

## Continuation from Assignment I

| Assignment I Component | Assignment II Module |
|------------------------|---------------------|
| `EpisodeLog` dataclass | `entities.py` |
| `WarehouseGrid` class | `grid.py` |
| `RobotSimulator` class | `simulator.py` |
| `bfs_shortest_path()` | `planner.py` |
| `greedy_random_path()` | `planner.py` |
| `execute_path()` | `inference.py` |
| `run_random_task_and_log()` | `inference.py` |
| `collect_logs()` F1 | `data_ingestion.py` |
| `clean_episodes()` F2 | `data_ingestion.py` |
| `engineer_features()` F3 | `feature_engineering.py` |
| `compute_labels()` F4 | `feature_engineering.py` |
| `split_dataset()` F5 | `training.py` |
| `score_path()` | `navigation_model.py` |
| `plan_path_with_model()` | `navigation_model.py` |
| `main_demo()` | `demo.py` |
