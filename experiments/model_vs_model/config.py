"""
Configuration for model-vs-model evaluation.
"""

GAMES_PATH = Path(__file__).resolve().parent().parent().parent()

AVAILABLE_MODELS = {
    "qwen-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen-8b": "Qwen/Qwen3-8B-Instruct",
    "qwen-14b": "Qwen/Qwen3-14B-Instruct",
    "qwen-32b": "Qwen/Qwen3-32B-Instruct",
}

DEFAULT_NUM_GAMES = 100
DEFAULT_NUM_ROUNDS = 5
DEFAULT_TEMPERATURE = 0.7  # same as inference
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_MAX_SEQ_LENGTH = 768

SYSTEM_PROMPT = (
    "You are playing Kuhn Poker.\n"
    "Respond with EXACTLY ONE action token and NOTHING ELSE.\n"
    "Valid outputs: [check] or [bet] or [call] or [fold].\n"
    "Do not add any whitespace, punctuation, explanation, or extra text.\n"
)