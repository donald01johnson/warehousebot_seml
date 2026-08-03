"""
Configuration constants for the Warehouse Robot Navigation System.

Assignment II - AIML ZG535, BITS Pilani WILP
Group 101
"""

# Grid movement directions: right, left, down, up
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Simulation constants
TIME_PER_STEP = 1.0  # seconds per step

# Default grid parameters
DEFAULT_ROWS = 10
DEFAULT_COLS = 10
DEFAULT_OBSTACLE_PROB = 0.2

# Dataset split ratios
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# Simulation parameters
DEFAULT_NUM_EPISODES = 200
DEFAULT_NUM_CANDIDATES = 5
MAX_GREEDY_STEPS = 200

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 8000
