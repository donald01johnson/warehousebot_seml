"""
Dataset splitting and ML model training — F5 and Assignment II ML requirement.

Implements:
    F5  – Dataset splitting  (Pipes-and-Filters, Assignment I continuation)
    ML  – RandomForest / LinearRegression training and evaluation
          (Assignment II Requirements 7 and 8)

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 101
"""

import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO
from .logging_config import get_logger

logger = get_logger("training")

# Feature columns used for ML training
FEATURE_COLUMNS = [
    "path_len",
    "shortest_path_len",
    "obstacle_density",
    "collisions",
    "replans",
    "path_optimality_ratio",
    "time_per_step",
]

TARGET_COLUMN = "label_travel_time"


def split_dataset(
    episodes: List[Dict],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    F5 – Dataset splitting.

    Shuffles and partitions episodes into train, validation, and test sets.

    Args:
        episodes   : List of fully-labeled episode dictionaries.
        train_ratio: Fraction for training   (default 0.70).
        val_ratio  : Fraction for validation (default 0.15).
                     Remaining fraction goes to test.

    Returns:
        Tuple (train, val, test) of episode lists.

    Raises:
        ValueError: If train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        logger.error(
            "Invalid split ratios: train=%.2f + val=%.2f >= 1.0",
            train_ratio,
            val_ratio,
        )
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0. "
            f"Got {train_ratio + val_ratio:.2f}."
        )

    if not episodes:
        logger.warning("split_dataset: empty episode list provided.")
        return [], [], []

    episodes = deepcopy(episodes)
    random.shuffle(episodes)

    n = len(episodes)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train = episodes[:n_train]
    val = episodes[n_train: n_train + n_val]
    test = episodes[n_train + n_val:]

    logger.info(
        "Dataset split: total=%d | train=%d | val=%d | test=%d",
        n,
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def _prepare_arrays(
    episodes: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert episode dicts into feature matrix X and label vector y.

    Skips any episode that is missing a required feature or label.

    Args:
        episodes: List of episode dictionaries with features and labels.

    Returns:
        Tuple (X, y) of numpy float arrays.
    """
    X, y = [], []

    for ep in episodes:
        try:
            row = [float(ep[col]) for col in FEATURE_COLUMNS]
            label = float(ep[TARGET_COLUMN])
            X.append(row)
            y.append(label)
        except KeyError as exc:
            logger.warning(
                "_prepare_arrays: skipping episode — missing key %s.", str(exc)
            )

    return np.array(X, dtype=float), np.array(y, dtype=float)


def train_model(
    train_episodes: List[Dict],
    model_type: str = "random_forest",
) -> object:
    """
    Train an ML model to predict robot travel time from episode features.

    Satisfies Assignment II Requirement 7 (ML training) and
    Requirement 8 (model quality via MAE on the training set).

    Args:
        train_episodes: List of training episode dicts with features + labels.
        model_type    : 'random_forest' or 'linear_regression'.

    Returns:
        Fitted scikit-learn model object.

    Raises:
        ValueError   : If model_type is unsupported or train_episodes is empty.
        RuntimeError : If scikit-learn fitting fails.
    """
    if not train_episodes:
        logger.error("train_model: empty training set provided.")
        raise ValueError("Cannot train a model on an empty training set.")

    logger.info(
        "Starting model training: type=%s, samples=%d",
        model_type,
        len(train_episodes),
    )

    X, y = _prepare_arrays(train_episodes)

    if len(X) == 0:
        logger.error("train_model: no valid samples after feature extraction.")
        raise RuntimeError("No valid samples available for training.")

    try:
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif model_type == "linear_regression":
            model = LinearRegression()
        else:
            logger.error("Unsupported model_type: '%s'.", model_type)
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                "Choose 'random_forest' or 'linear_regression'."
            )

        model.fit(X, y)

        # Log training MAE for sanity check
        train_preds = model.predict(X)
        train_mae = mean_absolute_error(y, train_preds)
        logger.info(
            "Training complete: type=%s, n_samples=%d, train_MAE=%.4f",
            model_type,
            len(X),
            train_mae,
        )

        return model

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        logger.error("Model training failed: %s", str(exc), exc_info=True)
        raise RuntimeError(f"Model training failed: {exc}") from exc


def evaluate_model(
    model,
    episodes: List[Dict],
    split_name: str = "test",
) -> Dict:
    """
    Evaluate a trained model and compute model quality metrics.

    Satisfies Assignment II Requirement 8:
        - MAE  (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R²   (Coefficient of Determination)

    Args:
        model      : Fitted scikit-learn model.
        episodes   : Episode dicts for evaluation.
        split_name : Label for logging (e.g. 'val', 'test').

    Returns:
        Dict with keys: split, n_samples, mae, rmse, r2.
    """
    if not episodes:
        logger.warning("evaluate_model: empty %s set.", split_name)
        return {
            "split": split_name,
            "n_samples": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    X, y = _prepare_arrays(episodes)

    if len(X) == 0:
        logger.error("evaluate_model: no valid samples in %s set.", split_name)
        return {
            "split": split_name,
            "n_samples": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    try:
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2 = r2_score(y, preds)

        logger.info(
            "Evaluation [%s]: n=%d | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            split_name,
            len(X),
            mae,
            rmse,
            r2,
        )

        return {
            "split": split_name,
            "n_samples": len(X),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }

    except Exception as exc:
        logger.error(
            "evaluate_model failed on %s: %s",
            split_name,
            str(exc),
            exc_info=True
        )
        raise
