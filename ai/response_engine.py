"""AI Response Engine for development and future offline LLM integration."""

import datetime
import logging
import re
from typing import Protocol

logger = logging.getLogger("pi_assistant.ai")


class BaseResponseEngine(Protocol):
    """Protocol interface for response engines."""

    def generate_response(self, user_text: str) -> str:
        """Generate response text from user input."""
        ...


class DevelopmentResponseEngine:
    """Development rule-based response engine stub.

    Allows natural conversational testing without requiring heavy LLMs or API keys.
    Can later be swapped with offline LLM (e.g. llama.cpp, Ollama, ONNX) without altering core.
    """

    def __init__(self):
        self._patterns = [
            (r"\b(hello|hi|hey|greetings)\b", "Hello! How can I help you today?"),
            (r"\b(what can you do|who are you|what are you)\b", "I'm your assistant. I'm still being built, but I can hear and talk with you!"),
            (r"\b(time|what time)\b", self._get_time_response),
            (r"\b(joke|tell me a joke)\b", "Why do programmers prefer dark mode? Because light attracts bugs!"),
            (r"\b(how are you|how's it going)\b", "I'm doing wonderful, ready to assist!"),
            (r"\b(thank you|thanks)\b", "You're very welcome!"),
            (r"\b(bye|goodbye|see you)\b", "Goodbye! Have a fantastic day!"),
        ]

    def _get_time_response(self) -> str:
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    def generate_response(self, user_text: str) -> str:
        """Analyze text and generate response."""
        cleaned = user_text.strip()
        if not cleaned:
            return "I couldn't hear anything. Please try speaking again."

        lower = cleaned.lower()
        for pattern, response in self._patterns:
            if re.search(pattern, lower):
                if callable(response):
                    ans = response()
                else:
                    ans = response
                logger.info("[AI] Matched rule for '%s' -> '%s'", cleaned, ans)
                return ans

        # Fallback development response
        ans = f"I heard: {cleaned}. I am ready for the next command."
        logger.info("[AI] Fallback response for '%s' -> '%s'", cleaned, ans)
        return ans
