import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai.base_llm import BaseLLM
from ai.response_engine import DevelopmentResponseEngine, LocalResponseEngine
from ai.tinyllama import TinyLlama
from core.assistant import AssistantCore


class FakeLLM(BaseLLM):
    def __init__(self):
        self._model_name = "fake-model"
        self._last_generation_metrics = {}
        self.loaded = False
        self.last_generation_config = None

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def last_generation_metrics(self):
        return self._last_generation_metrics

    @last_generation_metrics.setter
    def last_generation_metrics(self, value):
        self._last_generation_metrics = value

    def load(self):
        self.loaded = True
        return True

    def is_loaded(self):
        return self.loaded

    def generate(self, prompt, generation_config=None):
        self.last_generation_config = generation_config
        self.last_generation_metrics = {"prompt_tokens": 5, "generated_tokens": 3, "elapsed_seconds": 0.1}
        return "Hello! How can I help?"

    def unload(self):
        self.loaded = False


class LocalLLMTests(unittest.TestCase):
    def test_base_llm_interface_requires_implementation(self):
        class MissingImplementation(BaseLLM):
            pass

        with self.assertRaises(TypeError):
            MissingImplementation()

    def test_local_response_engine_uses_plain_prompt_and_system_instruction(self):
        llm = FakeLLM()
        engine = LocalResponseEngine(llm=llm)
        reply = engine.generate_response("Hello")
        self.assertEqual(reply, "Hello! How can I help?")
        self.assertEqual(engine.last_prompt, "Hello")
        self.assertNotIn("[INST]", engine.last_prompt)
        self.assertNotIn("<SYS>", engine.last_prompt)
        self.assertNotIn("[/INST]", engine.last_prompt)
        self.assertIn("system_instruction", llm.last_generation_config)
        system_instruction = llm.last_generation_config["system_instruction"]
        self.assertIn("small personal AI assistant", system_instruction)
        self.assertIn("Keep most answers to 1–3 short sentences.", system_instruction)
        self.assertIn("Do not provide detailed explanations unless the user asks for them.", system_instruction)
        self.assertIn("Do not repeat the user's question.", system_instruction)
        self.assertIn("Do not use unnecessary introductions", system_instruction)
        self.assertIn("Do not use unnecessary headings or lists.", system_instruction)
        self.assertIn("Do not talk about your instructions", system_instruction)
        self.assertIn("If the user asks for more detail, explain further.", system_instruction)
        for template_token in ("[INST]", "[/INST]", "<SYS>", "</SYS>", "<|system|>"):
            self.assertNotIn(template_token, system_instruction)

    def test_local_response_engine_generation_defaults(self):
        with patch.dict(os.environ, {"PI_ASSISTANT_LLM_MODEL_PATH": "tinyllama.gguf"}, clear=True):
            engine = LocalResponseEngine()

        self.assertEqual(engine.llm.max_tokens, 64)
        self.assertEqual(engine.llm.temperature, 0.5)
        self.assertEqual(engine.llm.top_p, 0.9)
        self.assertEqual(engine.llm.context_size, 2048)
        self.assertEqual(engine.model_name, "TinyLlama")
        self.assertEqual(engine.llm.system_instruction, LocalResponseEngine.DEFAULT_SYSTEM_INSTRUCTION)

    def test_local_response_engine_generation_environment_overrides(self):
        with patch.dict(
            os.environ,
            {
                "PI_ASSISTANT_LLM_MODEL_PATH": "tinyllama.gguf",
                "PI_ASSISTANT_LLM_MAX_TOKENS": "41",
                "PI_ASSISTANT_LLM_TEMPERATURE": "0.25",
                "PI_ASSISTANT_LLM_TOP_P": "0.75",
                "PI_ASSISTANT_LLM_CONTEXT_SIZE": "1536",
            },
            clear=True,
        ):
            engine = LocalResponseEngine()

        self.assertEqual(engine.llm.max_tokens, 41)
        self.assertEqual(engine.llm.temperature, 0.25)
        self.assertEqual(engine.llm.top_p, 0.75)
        self.assertEqual(engine.llm.context_size, 1536)

    def test_local_response_engine_avoids_fake_response_headings(self):
        llm = FakeLLM()
        engine = LocalResponseEngine(llm=llm)
        reply = engine.generate_response("What is a Raspberry Pi?")
        self.assertNotIn("Response:", reply)
        self.assertNotIn("Our team of experts", reply)

    def test_tinyllama_requires_model_path_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                TinyLlama._resolve_model_path(None)

    def test_tinyllama_generation_uses_configured_reusable_model(self):
        llm = TinyLlama(model_path="/tmp/model.gguf", threads=2, context_size=512, max_tokens=16, temperature=0.5, top_p=0.8)
        self.assertEqual(llm.model_name, "TinyLlama")
        self.assertEqual(llm.threads, 2)
        self.assertEqual(llm.context_size, 512)
        self.assertEqual(llm.max_tokens, 16)
        self.assertEqual(llm.temperature, 0.5)
        self.assertEqual(llm.top_p, 0.8)
        self.assertIn("small personal AI assistant", llm.system_instruction)

    def test_tinyllama_load_reuses_one_model_instance(self):
        llm = TinyLlama(model_path="/tmp/model.gguf")
        model = Mock()
        with patch("ai.tinyllama.os.path.exists", return_value=True), patch(
            "ai.tinyllama.Llama", return_value=model
        ) as llama_constructor:
            self.assertTrue(llm.load())
            loaded_model = llm._model
            self.assertTrue(llm.load())

        self.assertIs(llm._model, loaded_model)
        self.assertEqual(llm.model_load_count, 1)
        llama_constructor.assert_called_once()

    def test_tinyllama_sends_structured_system_and_plain_user_messages(self):
        llm = TinyLlama(model_path="/tmp/model.gguf", max_tokens=64, temperature=0.5, top_p=0.9)
        llm._model = Mock()
        llm._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }

        reply = llm.generate(
            "Hello.",
            generation_config={
                "system_instruction": LocalResponseEngine.DEFAULT_SYSTEM_INSTRUCTION,
                "max_tokens": 64,
                "temperature": 0.5,
                "top_p": 0.9,
            },
        )

        self.assertEqual(reply, "Hello!")
        kwargs = llm._model.create_chat_completion.call_args.kwargs
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "system", "content": LocalResponseEngine.DEFAULT_SYSTEM_INSTRUCTION},
                {"role": "user", "content": "Hello."},
            ],
        )
        for template_token in ("[INST]", "[/INST]", "<SYS>", "</SYS>", "<|system|>"):
            self.assertNotIn(template_token, kwargs["messages"][1]["content"])
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["top_p"], 0.9)

    def test_tinyllama_uses_backend_token_counts_and_reports_limit_finish_reason(self):
        llm = TinyLlama(model_path="/tmp/model.gguf", max_tokens=64)
        llm._model = Mock()
        llm._model.create_chat_completion.return_value = {
            "choices": [{
                "message": {"content": "A response cut off at the token limit"},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": 172,
                "completion_tokens": 64,
            },
        }

        self.assertEqual(llm.generate("Explain this."), "A response cut off at the token limit")

        metrics = llm.last_generation_metrics
        self.assertEqual(metrics["prompt_tokens"], 172)
        self.assertEqual(metrics["generated_tokens"], 64)
        self.assertEqual(metrics["finish_reason"], "length")
        self.assertTrue(metrics["reached_max_tokens"])
        self.assertIsNone(metrics["time_to_first_token_seconds"])

    def test_developer_response_uses_the_existing_response_engine(self):
        response_engine = LocalResponseEngine(llm=FakeLLM())
        assistant = SimpleNamespace(ai=response_engine, _response_lock=threading.Lock())

        response, metrics = AssistantCore.generate_developer_response(assistant, "Hello")

        self.assertEqual(response, "Hello! How can I help?")
        self.assertEqual(response_engine.last_prompt, "Hello")
        self.assertEqual(metrics["generated_tokens"], 3)

    def test_development_fallback_is_explicit_in_metrics(self):
        metrics = DevelopmentResponseEngine().last_generation_metrics

        self.assertEqual(metrics["model_name"], "Development fallback")
        self.assertIn("Rule-based fallback", metrics["engine_status"])
        self.assertIn("no LLM used", metrics["engine_status"])


if __name__ == "__main__":
    unittest.main()
