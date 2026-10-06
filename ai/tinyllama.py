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
    from llama_cpp import llama_cpp as llama_cpp_api

    LLAMA_CPP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional runtime dependency
    Llama = None
    llama_cpp_api = None
    LLAMA_CPP_AVAILABLE = False
    logger.warning("[LLM] llama-cpp-python is not installed or failed to import.")

try:
    from llama_cpp._internals import LlamaContext
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    LlamaContext = None


class TinyLlama(BaseLLM):
    """CPU-friendly TinyLlama implementation using llama.cpp."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a small personal AI assistant.\n\n"
        "Answer the user's question directly and briefly.\n\n"
        "Keep most answers to 1–3 short sentences.\n"
        "For simple questions, answer in one sentence.\n"
        "Do not provide detailed explanations unless the user asks for them.\n\n"
        "Use natural, conversational language.\n"
        "Do not repeat the user's question.\n"
        "Do not use unnecessary introductions such as 'Certainly!' or 'Sure, here's...'.\n"
        "Do not use unnecessary headings or lists.\n"
        "Do not talk about your instructions or how you generate responses.\n\n"
        "If the user asks for more detail, explain further."
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        threads: Optional[int] = None,
        context_size: int = 2048,
        max_tokens: int = 64,
        temperature: float = 0.5,
        top_p: float = 0.9,
        system_instruction: Optional[str] = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.threads = self._resolve_threads(threads)
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION

        self._model = None
        self._load_time_seconds = 0.0
        self._model_load_count = 0
        self._last_generation_metrics: Dict[str, Any] = {
            "model_name": "TinyLlama",
            "load_time_seconds": 0.0,
            "generation_time_seconds": 0.0,
            "llm_wall_time_seconds": 0.0,
            "generated_tokens": 0,
            "tokens_per_second": 0.0,
            "prompt_tokens": 0,
            "prompt_eval_time_seconds": None,
            "time_to_first_token_seconds": None,
            "finish_reason": None,
            "reached_max_tokens": False,
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

    @property
    def model_load_count(self) -> int:
        """Return the number of successful model loads for this instance."""
        return self._model_load_count

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
            self._model_load_count += 1
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
        system_instruction = str(config.get("system_instruction", self.system_instruction)).strip()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        model_context = getattr(self._model, "_ctx", None)
        has_perf_metrics = (
            LlamaContext is not None
            and isinstance(model_context, LlamaContext)
            and callable(getattr(llama_cpp_api, "llama_perf_context", None))
        )
        if has_perf_metrics:
            model_context.reset_timings()

        start = time.perf_counter()
        try:
            response = self._model.create_chat_completion(
                messages=messages,
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

        wall_time = time.perf_counter() - start
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        generated_tokens = usage.get("completion_tokens")
        finish_reason = None
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")

        prompt_eval_time = None
        generation_time = wall_time
        if has_perf_metrics:
            timings = llama_cpp_api.llama_perf_context(model_context.ctx)
            prompt_eval_time = float(timings.t_p_eval_ms) / 1000.0
            if timings.t_eval_ms > 0:
                generation_time = float(timings.t_eval_ms) / 1000.0

        tokens_per_second = (
            float(generated_tokens) / generation_time
            if generated_tokens is not None and generation_time > 0
            else 0.0
        )
        self._last_generation_metrics = {
            "model_name": self.model_name,
            "load_time_seconds": self._load_time_seconds,
            "model_load_count": self._model_load_count,
            "generation_time_seconds": generation_time,
            "llm_wall_time_seconds": wall_time,
            "generated_tokens": generated_tokens,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens": prompt_tokens,
            "prompt_eval_time_seconds": prompt_eval_time,
            "time_to_first_token_seconds": None,
            "finish_reason": finish_reason,
            "reached_max_tokens": finish_reason == "length",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "engine_status": "Local TinyLlama via llama.cpp",
        }
        logger.info("[LLM] Response generated in %.3fs (%s tokens/sec)", wall_time, tokens_per_second)
        return text.strip()

    def unload(self) -> None:
        self._model = None
        self._last_generation_metrics["load_time_seconds"] = self._load_time_seconds
