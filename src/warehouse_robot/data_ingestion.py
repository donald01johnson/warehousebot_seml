"""
Data ingestion pipeline — F1 (collection) and F2 (cleaning).

Implements the first two filters of the Pipes-and-Filters architecture
carried forward from Assignment I.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 101
"""

from dataclasses import asdict
from typing import Dict, List

from .entities import EpisodeLog
from .logging_config import get_logger

logger = get_logger("data_ingestion")


def collect_logs(raw_logs: List[EpisodeLog]) -> List[Dict]:
    """
    F1 – Data collection: convert EpisodeLog objects to plain dicts.

    Args:
        raw_logs: List of EpisodeLog instances from simulation.

    Returns:
        List of episode dictionaries suitable for downstream processing.
    """
    if not raw_logs:
        logger.warning("collect_logs: received empty raw_logs list.")
        return []

    collected = [asdict(ep) for ep in raw_logs]

    logger.info(
        "collect_logs: converted %d EpisodeLog objects to dicts.",
        len(collected),
    )
    return collected


def clean_episodes(episodes: List[Dict]) -> List[Dict]:
    """
    F2 – Cleaning and validation: drop invalid or incomplete episodes.

    Removal criteria:
        - shortest_path_len == 0  → no valid path was found
        - travel_time <= 0        → invalid or zero-duration episode

    Args:
        episodes: Raw list of episode dictionaries from collect_logs().

    Returns:
        Filtered list containing only valid episodes.
    """
    if not episodes:
        logger.warning("clean_episodes: received empty episode list.")
        return []

    initial_count = len(episodes)
    cleaned: List[Dict] = []
    dropped = 0

    for ep in episodes:
        try:
            if ep.get("shortest_path_len", 0) == 0:
                logger.warning(
                    "Dropping episode — no valid path (shortest_path_len=0): "
                    "start=%s, goal=%s",
                    ep.get("start"),
                    ep.get("goal"),
                )
                dropped += 1
                continue

            if ep.get("travel_time", 0.0) <= 0:
                logger.warning(
                    "Dropping episode — invalid travel_time=%.2f: "
                    "start=%s, goal=%s",
                    ep.get("travel_time", 0.0),
                    ep.get("start"),
                    ep.get("goal"),
                )
                dropped += 1
                continue

            cleaned.append(ep)

        except Exception as exc:
            logger.error(
                "Unexpected error while cleaning episode: %s", str(exc))
            dropped += 1

    logger.info(
        "clean_episodes: retained=%d / total=%d, dropped=%d.",
        len(cleaned),
        initial_count,
        dropped,
    )
    return cleaned
