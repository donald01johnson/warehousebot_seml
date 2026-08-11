"""
Feature engineering and label computation — F3 and F4.

Implements the middle two filters of the Pipes-and-Filters architecture
carried forward from Assignment I.

Assignment II - AIMLCZG546, BITS Pilani WILP
Group 212
"""

from typing import Dict, List

from .logging_config import get_logger

logger = get_logger("feature_engineering")


def engineer_features(episodes: List[Dict]) -> List[Dict]:
    """
    F3 – Feature engineering: derive additional columns from raw fields.

    New features added:
        path_len              : actual number of steps taken
        path_optimality_ratio : path_len / shortest_path_len
        time_per_step         : travel_time / path_len

    Args:
        episodes: List of cleaned episode dictionaries.

    Returns:
        Same list enriched with new feature keys.
    """
    if not episodes:
        logger.warning("engineer_features: received empty episode list.")
        return []

    logger.info("Applying feature engineering to %d episodes.", len(episodes))

    for ep in episodes:
        try:
            path_len = len(ep.get("path", []))
            ep["path_len"] = path_len

            shortest = ep.get("shortest_path_len", 0)
            ep["path_optimality_ratio"] = (
                path_len / shortest if shortest > 0 else 1.0
            )

            ep["time_per_step"] = (
                ep["travel_time"] / path_len if path_len > 0 else 0.0
            )

        except Exception as exc:
            logger.error(
                "Feature engineering error for episode start=%s: %s",
                ep.get("start"),
                str(exc),
            )

    logger.info("Feature engineering complete.")
    return episodes


def compute_labels(episodes: List[Dict]) -> List[Dict]:
    """
    F4 – Label and metric computation.

    Labels added:
        label_travel_time : regression target (float)
        label_collisions  : collision count (int)
        is_safe           : 1 if collisions == 0, else 0 (binary)

    Args:
        episodes: List of feature-engineered episode dictionaries.

    Returns:
        Same list enriched with label keys.
    """
    if not episodes:
        logger.warning("compute_labels: received empty episode list.")
        return []

    logger.info("Computing labels for %d episodes.", len(episodes))

    for ep in episodes:
        try:
            ep["label_travel_time"] = ep["travel_time"]
            ep["label_collisions"] = ep["collisions"]
            ep["is_safe"] = 1 if ep["collisions"] == 0 else 0

        except Exception as exc:
            logger.error(
                "Label computation error for episode start=%s: %s",
                ep.get("start"),
                str(exc),
            )

    logger.info("Label computation complete.")
    return episodes
