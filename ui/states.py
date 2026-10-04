"""State definitions for the AI Assistant."""

from enum import Enum, auto


class AssistantState(Enum):
    """Core state system for the AI Assistant.

    Supported States:
        - IDLE: Calm resting state with smiling face and subtle automatic blinking.
        - LISTENING: Attentive expression waiting for / capturing speech.
        - THINKING: Animated pondering expression while processing AI responses.
        - SPEAKING: Animated talking mouth cycling through speech shapes.
        - ERROR: Alert / error visual state with automatic recovery to IDLE.
    """

    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()
