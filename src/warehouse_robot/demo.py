"""
Main demo script — full Assignment II pipeline end-to-end.

Demonstrates:
    1. Grid creation
    2. Episode simulation
    3. Data pipeline (F1-F5)
    4. ML model training and evaluation
    5. Model quality metrics
    6. Data quality metrics
    7. ML-based path planning

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

from .config import (
    DEFAULT_COLS,
    DEFAULT_NUM_EPISODES,
    DEFAULT_OBSTACLE_PROB,
    DEFAULT_ROWS,
)
from .data_ingestion import clean_episodes, collect_logs
from .feature_engineering import compute_labels, engineer_features
from .grid import WarehouseGrid
from .inference import RAW_LOGS, run_random_task_and_log
from .logging_config import setup_logging
from .navigation_model import plan_path_with_model
from .training import evaluate_model, split_dataset, train_model


def main_demo():
    """Run the complete warehouse robot navigation pipeline demo."""

    setup_logging()

    print("=" * 65)
    print("  WAREHOUSE ROBOT NAVIGATION SYSTEM - ASSIGNMENT II DEMO")
    print("  AIMLCZG546 | BITS Pilani WILP | Group 212")
    print("=" * 65)

    # Step 1: Create Grid
    print("\n[1] Initialising Warehouse Grid ...")
    grid = WarehouseGrid(
        rows=DEFAULT_ROWS,
        cols=DEFAULT_COLS,
        obstacle_prob=DEFAULT_OBSTACLE_PROB,
    )
    grid.display()
    print(f"    Grid size        : {DEFAULT_ROWS} x {DEFAULT_COLS}")
    print(f"    Obstacle density : {grid.obstacle_density():.3f}")

    # Step 2: Simulate Episodes
    print(f"\n[2] Simulating {DEFAULT_NUM_EPISODES} navigation episodes ...")
    RAW_LOGS.clear()

    for _ in range(DEFAULT_NUM_EPISODES):
        run_random_task_and_log(grid)

    print(f"    Collected : {len(RAW_LOGS)} raw episodes")

    # Step 3: Data Pipeline (F1-F5)
    print("\n[3] Running Data Pipeline ...")

    episodes = collect_logs(RAW_LOGS)
    print(f"    F1 Collected  : {len(episodes)} episodes")

    episodes = clean_episodes(episodes)
    print(f"    F2 Cleaned    : {len(episodes)} episodes")

    episodes = engineer_features(episodes)
    print("    F3 Features   : engineered")

    episodes = compute_labels(episodes)
    print("    F4 Labels     : computed")

    train, val, test = split_dataset(episodes)
    n_tr, n_va, n_te = len(train), len(val), len(test)
    print(f"    F5 Split: train={n_tr} | val={n_va} | test={n_te}")

    # Step 4: ML Training
    print("\n[4] Training RandomForest Model ...")
    model = train_model(train, model_type="random_forest")

    val_metrics = evaluate_model(
        model, val, split_name="validation"
    )
    test_metrics = evaluate_model(
        model, test, split_name="test"
    )

    print("\n    -- Model Quality Metrics --")
    print(f"    Validation  MAE  : {val_metrics['mae']}")
    print(f"    Validation  RMSE : {val_metrics['rmse']}")
    print(f"    Validation  R2   : {val_metrics['r2']}")
    print(f"    Test        MAE  : {test_metrics['mae']}")
    print(f"    Test        RMSE : {test_metrics['rmse']}")
    print(f"    Test        R2   : {test_metrics['r2']}")

    # Step 5: Data Quality Metrics
    print("\n[5] Data Quality Metrics ...")

    total = len(RAW_LOGS)
    valid = len(episodes)
    invalid_rate = (total - valid) / total if total > 0 else 0.0

    missing_count = sum(
        1
        for ep in episodes
        if any(ep.get(k) is None for k in ep)
    )
    missing_rate = missing_count / valid if valid > 0 else 0.0

    print(f"    Total episodes collected : {total}")
    print(f"    Valid episodes retained  : {valid}")
    print(f"    Invalid episode rate     : {invalid_rate:.4f}")
    print(f"    Missing value rate       : {missing_rate:.4f}")

    # Step 6: ML-Based Path Planning Demo
    print("\n[6] ML-Based Path Planning Demo ...")

    start = (0, 0)
    goal = (DEFAULT_ROWS - 1, DEFAULT_COLS - 1)

    if grid.is_free(start) and grid.is_free(goal):
        best_path = plan_path_with_model(grid, start, goal)
        print(f"    Route   : {start} -> {goal}")
        print(f"    Length  : {len(best_path)} steps")
        suffix = "..." if len(best_path) > 6 else ""
        print(f"    Path    : {best_path[:6]}{suffix}")
        print("\n    Grid with planned endpoints:")
        grid.display(start=start, goal=goal)
    else:
        print("    Start or goal cell is blocked in this grid.")

    print("=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main_demo()
