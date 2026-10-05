"""Abstract interface for local language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLLM(ABC):
    """Common interface for local, replaceable language model implementations."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable model identifier."""

    @model_name.setter
    @abstractmethod
    def model_name(self, value: str) -> None:
        """Update the model identifier."""

    @property
    @abstractmethod
    def last_generation_metrics(self) -> Dict[str, Any]:
        """Return telemetry from the most recent generation attempt."""

    @last_generation_metrics.setter
    @abstractmethod
    def last_generation_metrics(self, value: Dict[str, Any]) -> None:
        """Update telemetry from the most recent generation attempt."""

    @abstractmethod
    def load(self) -> bool:
        """Load the underlying model into memory and return True on success."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the model is currently loaded and ready."""

    @abstractmethod
    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate text conditioned on the provided prompt."""

    @abstractmethod
    def unload(self) -> None:
        """Release the loaded model resources."""
