"""Assistant interaction event definitions."""

from dataclasses import dataclass
from typing import Optional
from ui.states import AssistantState


@dataclass
class AssistantEvent:
    """Base event for assistant system notifications."""
    state: AssistantState
    message: Optional[str] = None


@dataclass
class SpeechRecognizedEvent:
    """Event triggered when speech is transcribed into text."""
    text: str


@dataclass
class ResponseGeneratedEvent:
    """Event triggered when AI response is generated."""
    prompt: str
    response: str


@dataclass
class AssistantErrorEvent:
    """Event triggered when an error occurs in any subsystem."""
    source: str
    error_message: str
