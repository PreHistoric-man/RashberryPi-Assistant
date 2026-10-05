"""TinyLlama local LLM wrapper backed by llama.cpp."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from ai.base_llm import BaseLLM

logger = logging.getLogger("pi_assistant.ai.tinyllama")

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional runtime dependency
    Llama = None
    LLAMA_CPP_AVAILABLE = False
    logger.warning("[LLM] llama-cpp-python is not installed or failed to import.")


class TinyLlama(BaseLLM):
    """CPU-friendly TinyLlama implementation using llama.cpp."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threads: Optional[int] = None,
        context_size: int = 2048,
        max_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.8,
        system_instruction: Optional[str] = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.threads = self._resolve_threads(threads)
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_instruction = system_instruction or (
            "You are a concise personal assistant. Answer the user's question directly in 1-3 short sentences. "
            "Be natural and conversational. Do not add introductions, headings, marketing language, or unrelated filler. "
            "When the user asks for detail, provide a brief but complete answer."
        )

        self._model = None
        self._load_time_seconds = 0.0
        self._last_generation_metrics: Dict[str, Any] = {
            "model_name": "TinyLlama",
            "load_time_seconds": 0.0,
            "generation_time_seconds": 0.0,
            "generated_tokens": 0,
            "tokens_per_second": 0.0,
            "prompt_tokens": 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @property
    def model_name(self) -> str:
        return "TinyLlama"

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value

    @property
    def last_generation_metrics(self) -> Dict[str, Any]:
        return dict(self._last_generation_metrics)

    @last_generation_metrics.setter
    def last_generation_metrics(self, value: Dict[str, Any]) -> None:
        self._last_generation_metrics = dict(value)

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        configured = (model_path or os.getenv("PI_ASSISTANT_LLM_MODEL_PATH") or "").strip()
        if not configured:
            raise ValueError(
                "TinyLlama model path is not configured. Set PI_ASSISTANT_LLM_MODEL_PATH to the GGUF file path."
            )
        return configured

    @staticmethod
    def _resolve_threads(threads: Optional[int]) -> int:
        if threads is not None:
            if threads < 1:
                raise ValueError("llm threads must be at least 1")
            return threads

        configured = (os.getenv("PI_ASSISTANT_LLM_THREADS") or "3").strip()
        try:
            resolved = int(configured)
        except ValueError:
            logger.warning("[LLM] Invalid PI_ASSISTANT_LLM_THREADS=%r; using 3", configured)
            return 3
        if resolved < 1:
            logger.warning("[LLM] Invalid thread count %s; using 3", resolved)
            return 3
        return resolved

    def load(self) -> bool:
        if self._model is not None:
            return True
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is required for TinyLlama; install it with pip install llama-cpp-python.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"TinyLlama GGUF model not found at '{self.model_path}'. Set PI_ASSISTANT_LLM_MODEL_PATH to the correct GGUF file."
            )

        start = time.perf_counter()
        try:
            logger.info("[LLM] Loading TinyLlama model from '%s' with %s threads", self.model_path, self.threads)
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=self.threads,
                verbose=False,
            )
            self._load_time_seconds = time.perf_counter() - start
            self._last_generation_metrics["load_time_seconds"] = self._load_time_seconds
            logger.info("[LLM] TinyLlama model loaded in %.3fs", self._load_time_seconds)
            return True
        except Exception as exc:  # pragma: no cover - depends on runtime backend
            self._model = None
            raise RuntimeError(f"TinyLlama model failed to load: {exc}") from exc

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
        if not prompt or not str(prompt).strip():
            return ""

        if not self.is_loaded():
            self.load()

        config = dict(generation_config or {})
        max_tokens = int(config.get("max_tokens", self.max_tokens))
        temperature = float(config.get("temperature", self.temperature))
        top_p = float(config.get("top_p", self.top_p))
        stream = bool(config.get("stream", False))

        start = time.perf_counter()
        try:
            response = self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            text = ""
            choices = response.get("choices", []) if isinstance(response, dict) else []
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                text = str(message.get("content", "")).strip()
            if not text and isinstance(response, dict):
                text = str(response.get("content", "")).strip()
            if not text and hasattr(response, "choices"):
                first_choice = response.choices[0]
                if hasattr(first_choice, "text"):
                    text = str(first_choice.text).strip()
            if not text:
                raise RuntimeError("TinyLlama returned an empty response body.")
        except Exception as exc:  # pragma: no cover - runtime backend specific
            raise RuntimeError(f"TinyLlama generation failed: {exc}") from exc

        elapsed = time.perf_counter() - start
        prompt_tokens = max(1, len(prompt.split()))
        generated_tokens = max(1, len(text.split()))
        tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0
        self._last_generation_metrics = {
            "model_name": self.model_name,
            "load_time_seconds": self._load_time_seconds,
            "generation_time_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens": prompt_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        logger.info("[LLM] Response generated in %.3fs (%s tokens/sec)", elapsed, tokens_per_second)
        return text.strip()

    def unload(self) -> None:
        self._model = None
        self._last_generation_metrics["load_time_seconds"] = self._load_time_seconds
