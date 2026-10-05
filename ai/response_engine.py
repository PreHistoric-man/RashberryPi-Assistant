"""AI response engines for offline local LLM integration and development fallback."""

import datetime
import logging
import os
import re
from typing import Any, Dict, Optional, Protocol

from ai.base_llm import BaseLLM
from ai.tinyllama import TinyLlama

logger = logging.getLogger("pi_assistant.ai")


class BaseResponseEngine(Protocol):
    """Protocol interface for response engines."""

    def generate_response(self, user_text: str) -> str:
        """Generate response text from user input."""
        ...


class DevelopmentResponseEngine:
    """Development rule-based response engine stub."""

    @property
    def model_name(self) -> str:
        return "development"

    @property
    def last_generation_metrics(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "generated_tokens": 0, "generation_time_seconds": 0.0}

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
        self.last_prompt: str = ""
        self.last_response: str = ""

    def _get_time_response(self) -> str:
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    def generate_response(self, user_text: str) -> str:
        cleaned = user_text.strip()
        self.last_prompt = cleaned
        if not cleaned:
            self.last_response = "I couldn't hear anything. Please try speaking again."
            return self.last_response

        lower = cleaned.lower()
        for pattern, response in self._patterns:
            if re.search(pattern, lower):
                if callable(response):
                    ans = response()
                else:
                    ans = response
                self.last_response = ans
                logger.info("[AI] Matched rule for '%s' -> '%s'", cleaned, ans)
                return ans

        ans = f"I heard: {cleaned}. I am ready for the next command."
        self.last_response = ans
        logger.info("[AI] Fallback response for '%s' -> '%s'", cleaned, ans)
        return ans


class LocalResponseEngine:
    """Application-level interface that adapts a local LLM to the assistant pipeline."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a concise personal assistant. Answer the user's question directly in 1-3 short sentences. "
        "Be natural and conversational. Do not add introductions, headings, marketing language, or unrelated filler. "
        "When the user asks for detail, provide a brief but complete answer."
    )

    def __init__(self, llm: Optional[BaseLLM] = None, system_instruction: Optional[str] = None):
        self.llm = llm or TinyLlama(
            model_path=os.getenv("PI_ASSISTANT_LLM_MODEL_PATH"),
            threads=int(os.getenv("PI_ASSISTANT_LLM_THREADS", "3") or "3"),
            context_size=int(os.getenv("PI_ASSISTANT_LLM_CONTEXT_SIZE", "2048") or "2048"),
            max_tokens=int(os.getenv("PI_ASSISTANT_LLM_MAX_TOKENS", "64") or "64"),
            temperature=float(os.getenv("PI_ASSISTANT_LLM_TEMPERATURE", "0.3") or "0.3"),
            top_p=float(os.getenv("PI_ASSISTANT_LLM_TOP_P", "0.8") or "0.8"),
        )
        self.system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION
        self.last_prompt = ""
        self.last_response = ""
        self._last_generation_metrics: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return getattr(self.llm, "model_name", "local-llm")

    @property
    def last_generation_metrics(self) -> Dict[str, Any]:
        return self._last_generation_metrics or getattr(self.llm, "last_generation_metrics", {})

    def _build_prompt(self, user_text: str) -> str:
        return user_text.strip()

    def generate_response(self, user_text: str) -> str:
        prompt = self._build_prompt(user_text)
        self.last_prompt = prompt
        try:
            text = self.llm.generate(
                prompt,
                generation_config={
                    "max_tokens": getattr(self.llm, "max_tokens", 128),
                    "temperature": getattr(self.llm, "temperature", 0.7),
                    "top_p": getattr(self.llm, "top_p", 0.9),
                    "system_instruction": self.system_instruction,
                },
            )
        except Exception as exc:
            logger.error("[AI] Local LLM generation failed: %s", exc)
            raise RuntimeError(f"Local LLM generation failed: {exc}") from exc

        self.last_response = text.strip()
        self._last_generation_metrics = getattr(self.llm, "last_generation_metrics", {})
        return self.last_response


TinyLlamaResponseEngine = LocalResponseEngine
