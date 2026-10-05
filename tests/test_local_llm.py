import os
import unittest
from unittest.mock import Mock, patch

from ai.base_llm import BaseLLM
from ai.response_engine import LocalResponseEngine
from ai.tinyllama import TinyLlama


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
        self.assertIn("concise personal assistant", llm.last_generation_config["system_instruction"].lower())

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
        self.assertIn("concise personal assistant", llm.system_instruction.lower())

    def test_tinyllama_sends_plain_user_message_without_system_role(self):
        llm = TinyLlama(model_path="/tmp/model.gguf", max_tokens=64, temperature=0.3, top_p=0.8)
        llm._model = Mock()
        llm._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }

        reply = llm.generate(
            "Hello.",
            generation_config={
                "system_instruction": "You are a concise personal assistant.",
                "max_tokens": 64,
                "temperature": 0.3,
                "top_p": 0.8,
            },
        )

        self.assertEqual(reply, "Hello!")
        kwargs = llm._model.create_chat_completion.call_args.kwargs
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Hello."}])
        self.assertNotIn("[INST]", kwargs["messages"][0]["content"])
        self.assertNotIn("<SYS>", kwargs["messages"][0]["content"])
        self.assertNotIn("<|system|>", kwargs["messages"][0]["content"])
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["top_p"], 0.8)


if __name__ == "__main__":
    unittest.main()
