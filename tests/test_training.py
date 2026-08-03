"""
ML model training and inference tests.

Assignment II Requirement 7 - ML Training & Inference Tests
Assignment II Requirement 8 - Model Quality Metrics (MAE, RMSE, R2)
Assignment II - AIML ZG535, BITS Pilani WILP | Group 101
"""

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
from src.warehouse_robot.training import (
    FEATURE_COLUMNS,
    _prepare_arrays,
    evaluate_model,
    split_dataset,
    train_model,
)


# --- Helpers ---


def make_episode(path_len=5, travel_time=5.0, collisions=0):
    path = [(i, 0) for i in range(path_len)]
    return EpisodeLog(
        start=(0, 0),
        goal=(path_len - 1, 0),
        path=path,
        shortest_path_len=path_len,
        travel_time=travel_time,
        collisions=collisions,
        replans=0,
        obstacle_density=0.1,
    )


def build_labeled(n=80):
    """Generate n labeled episodes for training."""
    import random

    raw = []
    for _ in range(n):
        pl = random.randint(3, 15)
        raw.append(make_episode(path_len=pl, travel_time=float(pl)))
    collected = collect_logs(raw)
    cleaned = clean_episodes(collected)
    featured = engineer_features(cleaned)
    return compute_labels(featured)


# --- Training tests ---


class TestModelTraining:

    def test_random_forest_trains_without_error(self):
        """Model must return a fitted object without raising."""
        labeled = build_labeled(80)
        train, _, _ = split_dataset(labeled)
        model = train_model(train, model_type="random_forest")
        assert model is not None

    def test_linear_regression_trains_without_error(self):
        labeled = build_labeled(60)
        train, _, _ = split_dataset(labeled)
        model = train_model(train, model_type="linear_regression")
        assert model is not None

    def test_unsupported_model_type_raises_value_error(self):
        labeled = build_labeled(40)
        with pytest.raises(ValueError, match="Unsupported model_type"):
            train_model(labeled, model_type="xgboost")

    def test_empty_training_set_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            train_model([], model_type="random_forest")

    def test_overfit_small_batch(self):
        """
        Overfit test: train and evaluate on the same set.
        Training MAE must be very small (< 0.5 seconds).
        This proves the model can learn from the data.
        """
        labeled = build_labeled(20)
        model = train_model(labeled, model_type="random_forest")
        metrics = evaluate_model(
            model, labeled, split_name="overfit_check"
        )
        assert metrics["mae"] is not None
        assert metrics["mae"] < 0.5, (
            f"Expected overfit MAE < 0.5, got {metrics['mae']}."
        )

    def test_model_has_predict_method(self):
        labeled = build_labeled(40)
        train, _, _ = split_dataset(labeled)
        model = train_model(train)
        assert hasattr(model, "predict")


# --- Inference tests ---


class TestModelInference:

    def test_predictions_are_non_negative(self):
        """Travel time predictions must never be negative."""
        labeled = build_labeled(80)
        train, _, test = split_dataset(labeled)
        model = train_model(train)
        X, _ = _prepare_arrays(test)
        predictions = model.predict(X)
        assert all(p >= 0 for p in predictions), (
            f"Negative prediction found: min={min(predictions):.3f}"
        )

    def test_evaluation_returns_required_metric_keys(self):
        """evaluate_model must return mae, rmse, and r2."""
        labeled = build_labeled(80)
        train, val, _ = split_dataset(labeled)
        model = train_model(train)
        metrics = evaluate_model(model, val, split_name="validation")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_evaluation_mae_is_numeric(self):
        labeled = build_labeled(80)
        train, val, _ = split_dataset(labeled)
        model = train_model(train)
        metrics = evaluate_model(model, val)
        assert isinstance(metrics["mae"], float)
        assert metrics["mae"] >= 0

    def test_evaluation_rmse_ge_mae(self):
        """RMSE must always be >= MAE (mathematical property)."""
        labeled = build_labeled(80)
        train, val, _ = split_dataset(labeled)
        model = train_model(train)
        metrics = evaluate_model(model, val)
        assert metrics["rmse"] >= metrics["mae"], (
            "RMSE must be >= MAE. Mathematical invariant violated."
        )

    def test_directional_longer_path_higher_time(self):
        """
        Directional invariance: path of length 12 must predict
        higher travel time than path of length 3.
        """
        training_raw = []
        for pl in range(3, 16):
            for _ in range(8):
                training_raw.append(
                    make_episode(
                        path_len=pl, travel_time=float(pl)
                    )
                )

        collected = collect_logs(training_raw)
        cleaned = clean_episodes(collected)
        featured = engineer_features(cleaned)
        labeled = compute_labels(featured)

        model = train_model(labeled)

        short_raw = [make_episode(path_len=3, travel_time=3.0)]
        long_raw = [make_episode(path_len=12, travel_time=12.0)]

        def prep(raw_list):
            c = collect_logs(raw_list)
            cl = clean_episodes(c)
            fe = engineer_features(cl)
            la = compute_labels(fe)
            X, _ = _prepare_arrays(la)
            return X

        X_short = prep(short_raw)
        X_long = prep(long_raw)

        pred_short = model.predict(X_short)[0]
        pred_long = model.predict(X_long)[0]

        assert pred_long > pred_short, (
            f"Longer path must predict higher time. "
            f"short={pred_short:.3f}, long={pred_long:.3f}"
        )

    def test_evaluate_empty_set_returns_none_metrics(self):
        """Empty evaluation set must return None metrics."""
        labeled = build_labeled(30)
        model = train_model(labeled)
        metrics = evaluate_model(model, [])
        assert metrics["mae"] is None
        assert metrics["rmse"] is None
        assert metrics["r2"] is None


# --- _prepare_arrays helper ---


class TestPrepareArrays:

    def test_correct_feature_count(self):
        labeled = build_labeled(20)
        X, y = _prepare_arrays(labeled)
        assert X.shape[1] == len(FEATURE_COLUMNS)

    def test_correct_sample_count(self):
        labeled = build_labeled(20)
        X, y = _prepare_arrays(labeled)
        assert len(X) == len(y)

    def test_skips_missing_key_episode(self):
        labeled = build_labeled(10)
        broken = dict(labeled[0])
        del broken["path_len"]
        combined = labeled[1:] + [broken]
        X, y = _prepare_arrays(combined)
        assert len(X) == len(labeled) - 1
