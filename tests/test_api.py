"""
Integration tests for the FastAPI REST endpoints.

Assignment II Requirement 6 - Integration Tests
Assignment II - AIMLCZG546, BITS Pilani WILP | Group 212
"""

from fastapi.testclient import TestClient

from src.warehouse_robot.api import app

client = TestClient(app)

BASE_PAYLOAD = {
    "rows": 8,
    "cols": 8,
    "obstacle_probability": 0.0,
    "start": [0, 0],
    "goal": [7, 7],
    "num_candidates": 3,
}


# --- GET /health ---


class TestHealthEndpoint:

    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_healthy(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_service_name_present(self):
        data = client.get("/health").json()
        assert "service" in data


# --- POST /predict-path ---


class TestPredictPathEndpoint:

    def test_valid_request_returns_200(self):
        response = client.post("/predict-path", json=BASE_PAYLOAD)
        assert response.status_code == 200

    def test_response_status_is_success(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert data["status"] == "success"

    def test_path_is_list(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert isinstance(data["path"], list)

    def test_path_starts_at_start(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert data["path"][0] == [0, 0]

    def test_path_ends_at_goal(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert data["path"][-1] == [7, 7]

    def test_path_length_matches_path(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert data["path_length"] == len(data["path"])

    def test_score_is_positive(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert data["score"] > 0

    def test_message_present(self):
        data = client.post("/predict-path", json=BASE_PAYLOAD).json()
        assert "message" in data

    def test_invalid_grid_size_returns_422(self):
        payload = dict(BASE_PAYLOAD)
        payload["rows"] = 1
        response = client.post("/predict-path", json=payload)
        assert response.status_code == 422

    def test_invalid_obstacle_prob_returns_422(self):
        payload = dict(BASE_PAYLOAD)
        payload["obstacle_probability"] = 1.5
        response = client.post("/predict-path", json=payload)
        assert response.status_code == 422


# --- POST /score-path ---


class TestScorePathEndpoint:

    def test_valid_path_returns_200(self):
        payload = {
            "rows": 8,
            "cols": 8,
            "obstacle_probability": 0.0,
            "path": [[0, 0], [0, 1], [0, 2], [0, 3]],
        }
        response = client.post("/score-path", json=payload)
        assert response.status_code == 200

    def test_score_is_returned(self):
        payload = {
            "rows": 8,
            "cols": 8,
            "obstacle_probability": 0.0,
            "path": [[0, 0], [1, 0], [2, 0]],
        }
        data = client.post("/score-path", json=payload).json()
        assert "score" in data
        assert data["score"] >= 0

    def test_longer_path_higher_score(self):
        """Longer path must receive higher or equal score."""
        base = {
            "rows": 8,
            "cols": 8,
            "obstacle_probability": 0.0,
        }
        short_payload = {
            **base,
            "path": [[0, 0], [1, 0], [2, 0]],
        }
        long_payload = {
            **base,
            "path": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
        }

        r_short = client.post("/score-path", json=short_payload).json()
        r_long = client.post("/score-path", json=long_payload).json()
        score_short = r_short["score"]
        score_long = r_long["score"]

        assert score_long >= score_short


# --- POST /simulate ---


class TestSimulateEndpoint:

    def test_valid_request_returns_200(self):
        response = client.post("/simulate", json=BASE_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_travel_time(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert "travel_time" in data
        assert data["travel_time"] > 0

    def test_response_contains_collisions(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert "collisions" in data
        assert data["collisions"] >= 0

    def test_obstacle_density_in_range(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert 0.0 <= data["obstacle_density"] <= 1.0

    def test_path_starts_at_start(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert data["path"][0] == [0, 0]

    def test_path_ends_at_goal(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert data["path"][-1] == [7, 7]

    def test_status_is_success(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert data["status"] == "success"

    def test_path_length_field_correct(self):
        data = client.post("/simulate", json=BASE_PAYLOAD).json()
        assert data["path_length"] == len(data["path"])
