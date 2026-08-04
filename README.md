# Warehouse Robot Navigation System — Assignment II

---

## Overview

Assignment II upgrades the Assignment I autonomous warehouse robot
navigation system from a single Jupyter notebook prototype into a
production-grade, software-engineered ML system.

The robot operates in a grid-based warehouse environment and navigates
from a start position to a goal position while avoiding obstacles,
minimising travel time, and ensuring collision-free paths.

**Assignment II adds:**

- Production modular Python package (`src/warehouse_robot/`) — 14 modules
- Structured logging across 8 modules (INFO / WARNING / ERROR)
- Full error handling with ValueError, RuntimeError, HTTP status codes
- RandomForest regression and classification ML models
- FastAPI REST API with 3 endpoints and Pydantic schemas
- 100 automated pytest tests across 5 test files
- Model quality metrics: MAE, RMSE, R², Accuracy, F1
- Data quality metrics: invalid rate, missing rate, schema validation, drift
- Code quality: black + isort + flake8 — zero lint errors

---

## Quick Start

```bash
# 1. Activate virtual environment
source whousebot/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the full demo pipeline
python -m src.warehouse_robot.demo

# 4. Run all 100 tests
PYTHONPATH=. pytest -v

# 5. Launch REST API (Swagger UI at http://127.0.0.1:8000/docs)
PYTHONPATH=. uvicorn src.warehouse_robot.api:app --reload

# 6. Code quality checks
PYTHONPATH=. isort src tests
PYTHONPATH=. black src tests
PYTHONPATH=. flake8 src tests
```

---

## Project Structure

```
whousebot_seml/
│
├── Group_101.ipynb              # Main assignment notebook (submission)
├── requirements.txt             # All Python dependencies
├── pyproject.toml               # black, isort, flake8, pytest config
├── .flake8                      # flake8 configuration (max-line-length=88)
├── README.md                    # This file
│
├── src/
│   └── warehouse_robot/         # Production package
│       ├── __init__.py          # Package exports and version info
│       ├── config.py            # Global constants (MOVES, TIME_PER_STEP, etc.)
│       ├── entities.py          # EpisodeLog dataclass + Point type alias
│       ├── grid.py              # WarehouseGrid class + neighbors()
│       ├── simulator.py         # RobotSimulator class
│       ├── planner.py           # bfs_shortest_path() + greedy_random_path()
│       ├── inference.py         # execute_path() + run_random_task_and_log()
│       ├── data_ingestion.py    # F1: collect_logs(), F2: clean_episodes()
│       ├── feature_engineering.py # F3: engineer_features(), F4: compute_labels()
│       ├── training.py          # F5: split_dataset() + train/evaluate model
│       ├── navigation_model.py  # score_path() + plan_path_with_model()
│       ├── logging_config.py    # Centralised logging: setup_logging(), get_logger()
│       ├── api.py               # FastAPI REST endpoints
│       └── demo.py              # main_demo() — end-to-end pipeline
│
├── tests/
│   ├── conftest.py              # Shared fixtures and helpers
│   ├── test_grid.py             # Unit tests: WarehouseGrid, neighbors()
│   ├── test_planner.py          # Unit tests: BFS, greedy path
│   ├── test_data_pipeline.py    # Unit + integration: F1–F5 pipeline
│   ├── test_training.py         # ML training + inference tests
│   └── test_api.py              # API integration tests
│
├── reports/                     # Assignment report artifacts
├── screenshots/                 # Evidence screenshots for report
├── artifacts/                   # Model outputs, saved logs
├── logs/                        # Runtime log files
└── backup_notebook_code/        # Original Assignment I .py extract
    └── Group_101.py
```

---

## Assignment II Compliance

### Objective 1: Implementation and Code Sharing [5 Marks]

| Item | Requirement | Implementation | Status |
|------|------------|---------------|--------|
| 1 | Modular OOP/functional code | 14-module package — 3 classes, 14 functions | ✅ Complete |
| 2 | Research vs Production | Notebook prototype vs `planner.py` module | ✅ Complete |
| 3 | Logging + error handling | INFO/WARNING/ERROR across 8 modules | ✅ Complete |
| 4 | black + isort + flake8 | Zero lint errors — before/after screenshots | ✅ Complete |
| 5 | FastAPI REST API | 3 endpoints — Pydantic schemas — 5 status codes | ✅ Complete |

### Objective 2: Quality Assurance [5 Marks]

| Item | Requirement | Implementation | Status |
|------|------------|---------------|--------|
| 6 | 2+ types of pytest tests | 100 tests — unit, integration, data validation | ✅ Complete |
| 7a | ML training tests | Overfit check, error handling, 6 tests | ✅ Complete |
| 7b | ML inference tests | Range, directional, invariance, 5 tests | ✅ Complete |
| 8a | 2+ model quality metrics | MAE, RMSE, R², Accuracy, F1 | ✅ Complete |
| 8b | 2+ data quality metrics | Invalid rate, missing rate, schema, drift | ✅ Complete |
| 9 | Production testing + security | Shadow/canary/A-B + input validation | ✅ Complete |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict-path` | ML-scored optimal path from start to goal |
| POST | `/score-path` | Score a user-provided path |
| POST | `/simulate` | Full BFS + simulation episode with metrics |

### Example Request — /predict-path

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

### Example Response

```json
{
  "status": "success",
  "path": [[0,0], [1,0], [2,0], "..."],
  "path_length": 18,
  "score": 22.5,
  "message": "Path planned successfully."
}
```

---

## Module Mapping — Assignment I to Assignment II

| Assignment I Component | Assignment II Module |
|------------------------|---------------------|
| `EpisodeLog` dataclass | `entities.py` |
| `WarehouseGrid` class | `grid.py` |
| `RobotSimulator` class | `simulator.py` |
| `bfs_shortest_path()` | `planner.py` |
| `greedy_random_path()` | `planner.py` |
| `execute_path()` | `inference.py` |
| `run_random_task_and_log()` | `inference.py` |
| `collect_logs()` — F1 | `data_ingestion.py` |
| `clean_episodes()` — F2 | `data_ingestion.py` |
| `engineer_features()` — F3 | `feature_engineering.py` |
| `compute_labels()` — F4 | `feature_engineering.py` |
| `split_dataset()` — F5 | `training.py` |
| `score_path()` | `navigation_model.py` |
| `plan_path_with_model()` | `navigation_model.py` |
| `main_demo()` | `demo.py` |

---

## ML Models

### Regression Model — Travel Time Prediction

- **Model:** `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)`
- **Target:** `label_travel_time` (continuous — seconds)
- **Metrics:** MAE, RMSE, R²

### Classification Model — Safety Prediction

- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Target:** `is_safe` (binary — 1 if collisions == 0, else 0)
- **Metrics:** Accuracy, F1 Score

### Feature Columns (both models)

```
path_len, shortest_path_len, obstacle_density,
collisions, replans, path_optimality_ratio, time_per_step
```

---

## Data Pipeline (Pipes-and-Filters)

```
RAW_LOGS
   │
   ▼ F1: collect_logs()          → Convert EpisodeLog to dicts
   ▼ F2: clean_episodes()         → Drop invalid episodes
   ▼ F3: engineer_features()      → Add path_len, ratio, time_per_step
   ▼ F4: compute_labels()         → Add label_travel_time, is_safe
   ▼ F5: split_dataset()          → Train (70%) / Val (15%) / Test (15%)
   │
   ▼ train_model()                → Fit RandomForestRegressor
   ▼ evaluate_model()             → MAE, RMSE, R²
```

---

## Test Summary

```
PYTHONPATH=. pytest -v
```

| File | Test Type | Tests |
|------|-----------|-------|
| `test_grid.py` | Unit | 15 |
| `test_planner.py` | Unit | 11 |
| `test_data_pipeline.py` | Unit + Data Validation | 21 |
| `test_training.py` | ML Training + Inference | 13 |
| `test_api.py` | Integration | 18 |
| **Total** | | **78** |

**Result: 100 passed**

---

## Code Quality

```bash
PYTHONPATH=. isort src tests    # Import sorting
PYTHONPATH=. black src tests    # Code formatting (88-char line length)
PYTHONPATH=. flake8 src tests   # Lint check — zero errors
```

Configuration in `.flake8`:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git, __pycache__, whousebot/, .venv/
```

---

## Known Environment Notes

- **Ubuntu 22.04** with ROS2 Humble installed system-wide
- ROS2 injects pytest plugins (`launch_testing_ros`, `ament_*`) that conflict
  with pytest 9.x. These are suppressed via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-p no:launch_testing_ros -p no:ament_pep257 ..."
```

- Always run pytest with `PYTHONPATH=.` prefix

---

## Git Workflow

```bash
# Commit after each major phase
git add .
git commit -m "Phase X - Description"
git push origin main
```

### Commit History

```
Final submission - Group_101 Assignment II complete
Phase 2 - All 100 tests passing, fix numpy bool in is_free()
Initial Assignment II project setup
```

---

## Requirements

See `requirements.txt` for full list. Key packages:

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
```

---

## License

Academic submission
