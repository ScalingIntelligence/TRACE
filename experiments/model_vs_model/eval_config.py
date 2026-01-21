"""
Configuration for model-vs-model evaluation.
"""
from pathlib import Path

GAMES_PATH = Path(__file__).resolve().parent.parent.parent

AVAILABLE_MODELS = {
    "qwen-4b": "Qwen/Qwen3-4B",
    "qwen-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen-8b": "Qwen/Qwen3-8B",
    "qwen-14b": "Qwen/Qwen3-14B",
    "qwen-30b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}



DEFAULT_NUM_GAMES = 100
DEFAULT_NUM_ROUNDS = 1
DEFAULT_TEMPERATURE = 0.7  # same as inference
DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_MAX_SEQ_LENGTH = 8000


# Thinking mode enabled - models can reason before answering
SYSTEM_PROMPT = (
    "You are playing Kuhn Poker.\n"
    "Think through your decision carefully, then provide your action.\n"
    "Valid actions: [check], [bet], [call], [fold]\n"
    "End your response with your chosen action in brackets.\n"
)


ENABLE_THINKING = True
if not ENABLE_THINKING:
    DEFAULT_MAX_NEW_TOKENS = 8
    SYSTEM_PROMPT = (
        "/no_think\n"
        "You are playing Kuhn Poker.\n"
        "Respond with EXACTLY ONE action token and NOTHING ELSE.\n"
        "Valid outputs: [check] or [bet] or [call] or [fold].\n"
        "Do not add any whitespace, punctuation, explanation, or extra text.\n"
    )
