"""
Unit and Integration tests for the full data pipeline.

Covers F1-F5 (collect, clean, engineer, label, split).

Assignment II Requirements 6 & 8 - Unit Tests + Data Quality Metrics
Assignment II - AIMLCZG546, BITS Pilani WILP | Group 101
"""

from dataclasses import asdict

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
from src.warehouse_robot.training import split_dataset


# --- Helpers ---


def make_episode(
    path_len=5,
    travel_time=5.0,
    collisions=0,
    replans=0,
    obstacle_density=0.1,
):
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


# --- F1: collect_logs ---


class TestCollectLogs:

    def test_empty_input_returns_empty(self):
        assert collect_logs([]) == []

    def test_returns_list_of_dicts(self):
        ep = make_episode()
        result = collect_logs([ep])
        assert isinstance(result[0], dict)

    def test_correct_count(self):
        eps = [make_episode() for _ in range(10)]
        assert len(collect_logs(eps)) == 10

    def test_required_keys_present(self):
        ep = make_episode()
        result = collect_logs([ep])[0]
        for key in [
            "start", "goal", "path", "travel_time", "collisions"
        ]:
            assert key in result, f"Key '{key}' missing"


# --- F2: clean_episodes ---


class TestCleanEpisodes:

    def test_empty_input_returns_empty(self):
        assert clean_episodes([]) == []

    def test_drops_zero_path_length(self):
        ep = asdict(make_episode())
        ep["shortest_path_len"] = 0
        assert clean_episodes([ep]) == []

    def test_drops_zero_travel_time(self):
        ep = asdict(make_episode())
        ep["travel_time"] = 0.0
        assert clean_episodes([ep]) == []

    def test_drops_negative_travel_time(self):
        ep = asdict(make_episode())
        ep["travel_time"] = -1.0
        assert clean_episodes([ep]) == []

    def test_keeps_valid_episode(self):
        ep = asdict(make_episode())
        result = clean_episodes([ep])
        assert len(result) == 1

    def test_mixed_valid_invalid(self):
        valid = asdict(make_episode())
        invalid = asdict(make_episode())
        invalid["shortest_path_len"] = 0
        result = clean_episodes([valid, invalid])
        assert len(result) == 1


# --- F3: engineer_features ---


class TestEngineerFeatures:

    def test_empty_input_returns_empty(self):
        assert engineer_features([]) == []

    def test_adds_path_len(self):
        ep = asdict(make_episode(path_len=5))
        result = engineer_features([ep])
        assert result[0]["path_len"] == 5

    def test_optimality_ratio_perfect_path(self):
        ep = asdict(make_episode(path_len=5))
        result = engineer_features([ep])
        assert result[0]["path_optimality_ratio"] == pytest.approx(1.0)

    def test_time_per_step_correct(self):
        ep = asdict(make_episode(path_len=5, travel_time=5.0))
        result = engineer_features([ep])
        assert result[0]["time_per_step"] == pytest.approx(1.0)

    def test_zero_path_len_handled(self):
        ep = asdict(make_episode())
        ep["path"] = []
        result = engineer_features([ep])
        assert result[0]["path_len"] == 0
        assert result[0]["time_per_step"] == 0.0


# --- F4: compute_labels ---


class TestComputeLabels:

    def test_empty_input_returns_empty(self):
        assert compute_labels([]) == []

    def test_adds_all_label_keys(self):
        ep = asdict(make_episode())
        result = compute_labels([ep])[0]
        assert "label_travel_time" in result
        assert "label_collisions" in result
        assert "is_safe" in result

    def test_safe_flag_no_collisions(self):
        ep = asdict(make_episode(collisions=0))
        result = compute_labels([ep])[0]
        assert result["is_safe"] == 1

    def test_safe_flag_with_collisions(self):
        ep = asdict(make_episode(collisions=3))
        result = compute_labels([ep])[0]
        assert result["is_safe"] == 0

    def test_label_travel_time_matches(self):
        ep = asdict(make_episode(travel_time=7.0))
        result = compute_labels([ep])[0]
        assert result["label_travel_time"] == pytest.approx(7.0)


# --- F5: split_dataset ---


class TestSplitDataset:

    def test_empty_returns_three_empty(self):
        t, v, te = split_dataset([])
        assert t == [] and v == [] and te == []

    def test_correct_split_counts(self):
        eps = [asdict(make_episode()) for _ in range(100)]
        train, val, test = split_dataset(
            eps, train_ratio=0.7, val_ratio=0.15
        )
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_data_leakage(self):
        """Train, val, and test sets must be disjoint."""
        eps = [asdict(make_episode()) for _ in range(30)]
        for i, ep in enumerate(eps):
            ep["_id"] = i
        train, val, test = split_dataset(eps)
        t_ids = {ep["_id"] for ep in train}
        v_ids = {ep["_id"] for ep in val}
        te_ids = {ep["_id"] for ep in test}
        assert not (t_ids & v_ids), "Train and val overlap!"
        assert not (t_ids & te_ids), "Train and test overlap!"
        assert not (v_ids & te_ids), "Val and test overlap!"

    def test_all_episodes_accounted_for(self):
        eps = [asdict(make_episode()) for _ in range(40)]
        train, val, test = split_dataset(eps)
        assert len(train) + len(val) + len(test) == 40

    def test_invalid_ratios_raises(self):
        eps = [asdict(make_episode()) for _ in range(10)]
        with pytest.raises(ValueError, match="train_ratio"):
            split_dataset(eps, train_ratio=0.8, val_ratio=0.3)


# --- Integration: full pipeline ---


class TestFullPipelineIntegration:

    def test_end_to_end_pipeline(self):
        """Integration: EpisodeLog to split datasets without loss."""
        raw_eps = [make_episode(path_len=i + 2) for i in range(60)]

        collected = collect_logs(raw_eps)
        assert len(collected) == 60

        cleaned = clean_episodes(collected)
        assert len(cleaned) <= 60

        featured = engineer_features(cleaned)
        assert all("path_len" in ep for ep in featured)
        assert all("path_optimality_ratio" in ep for ep in featured)

        labeled = compute_labels(featured)
        assert all("is_safe" in ep for ep in labeled)
        assert all("label_travel_time" in ep for ep in labeled)

        train, val, test = split_dataset(labeled)
        assert len(train) + len(val) + len(test) == len(labeled)

    def test_data_quality_metrics(self):
        """
        Assignment II Requirement 8 - Data Quality Metrics.

        Verifies invalid episode rate and missing value rate.
        """
        raw_eps = [make_episode() for _ in range(50)]

        for _ in range(10):
            raw_eps.append(
                EpisodeLog(
                    start=(0, 0),
                    goal=(1, 0),
                    path=[],
                    shortest_path_len=0,
                    travel_time=0.0,
                    collisions=0,
                    replans=0,
                    obstacle_density=0.1,
                )
            )

        collected = collect_logs(raw_eps)
        cleaned = clean_episodes(collected)

        total = len(collected)
        valid = len(cleaned)
        invalid_rate = (total - valid) / total

        assert invalid_rate == pytest.approx(10 / 60, abs=0.01)

        missing = sum(
            1 for ep in cleaned
            if any(ep.get(k) is None for k in ep)
        )
        missing_rate = missing / valid if valid > 0 else 0.0
        assert missing_rate == 0.0
