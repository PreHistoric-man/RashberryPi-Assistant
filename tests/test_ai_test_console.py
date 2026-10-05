"""Tests for the Developer Mode text-based AI test console."""

import threading
import time
import unittest
from unittest.mock import Mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ui.developer.ai_test_console import AITestConsole
from ui.developer.developer_controller import DeveloperController
from ui.states import AssistantState


class DeveloperAITestConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.addCleanup(self.app.processEvents)

    def wait_for_generation(self, console):
        deadline = time.monotonic() + 2
        while console.is_generating and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertFalse(console.is_generating, "console generation did not finish")

    def test_empty_prompt_is_rejected(self):
        generator = Mock()
        console = AITestConsole(generator)
        console.prompt_input.setPlainText(" \n\t ")

        self.assertFalse(console.submit_prompt())
        generator.assert_not_called()
        self.assertFalse(console.is_generating)

    def test_prompt_is_passed_and_response_and_metrics_are_displayed(self):
        generator = Mock(return_value=(
            "A Raspberry Pi is a small single-board computer.",
            {
                "model_name": "TinyLlama",
                "generation_time_seconds": 1.2345,
                "generated_tokens": 18,
                "tokens_per_second": 14.59,
            },
        ))
        console = AITestConsole(generator)
        prompt = "What is a Raspberry Pi?\nAnswer briefly."
        console.prompt_input.setPlainText(prompt)

        self.assertTrue(console.submit_prompt())
        self.wait_for_generation(console)

        generator.assert_called_once_with(prompt)
        self.assertEqual(
            console.response_output.toPlainText(),
            "A Raspberry Pi is a small single-board computer.",
        )
        metrics = console.metrics_label.text()
        self.assertIn("Model: TinyLlama", metrics)
        self.assertIn("Generation time: 1.234 s", metrics)
        self.assertIn("Generated tokens: 18", metrics)
        self.assertIn("Tokens/sec: 14.59", metrics)
        self.assertTrue(console.send_button.isEnabled())

    def test_generation_error_is_displayed_and_console_recovers(self):
        generator = Mock(side_effect=RuntimeError("model file is unavailable"))
        console = AITestConsole(generator)
        console.prompt_input.setPlainText("Hello")

        self.assertTrue(console.submit_prompt())
        self.wait_for_generation(console)

        output = console.response_output.toPlainText()
        self.assertIn("ERROR", output)
        self.assertIn("TinyLlama generation failed:", output)
        self.assertIn("model file is unavailable", output)
        self.assertTrue(console.send_button.isEnabled())
        self.assertFalse(console.is_generating)

    def test_console_rejects_overlapping_generations(self):
        entered = threading.Event()
        release = threading.Event()

        def generate(prompt):
            entered.set()
            release.wait(timeout=2)
            return "done", {"model_name": "TinyLlama"}

        console = AITestConsole(generate)
        console.prompt_input.setPlainText("first prompt")

        self.assertTrue(console.submit_prompt())
        self.assertTrue(entered.wait(timeout=1))
        self.assertFalse(console.submit_prompt())
        self.assertEqual(console.generation_status.text(), "Generating...")

        release.set()
        self.wait_for_generation(console)

    def test_ctrl_enter_sends_multiline_prompt(self):
        generator = Mock(return_value=("answer", {"model_name": "TinyLlama"}))
        console = AITestConsole(generator)
        console.prompt_input.setPlainText("first line\nsecond line")
        console.prompt_input.setFocus()

        QTest.keyClick(
            console.prompt_input,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.wait_for_generation(console)

        generator.assert_called_once_with("first line\nsecond line")
        self.assertEqual(console.response_output.toPlainText(), "answer")

    def test_developer_controller_shortcuts_still_dispatch(self):
        controller = DeveloperController()
        requested_states = []
        interactions = []
        controller.state_change_requested.connect(requested_states.append)
        controller.interaction_requested.connect(lambda: interactions.append(True))
        controller.set_mode(True)

        class KeyEvent:
            def __init__(self, key):
                self._key = key

            def key(self):
                return self._key

        self.assertTrue(controller.handle_key_press(KeyEvent(Qt.Key.Key_3)))
        self.assertEqual(requested_states, [AssistantState.THINKING])
        self.assertTrue(controller.handle_key_press(KeyEvent(Qt.Key.Key_T)))
        self.assertEqual(interactions, [True])
        self.assertTrue(controller.handle_key_press(KeyEvent(Qt.Key.Key_Escape)))
        self.assertFalse(controller.is_enabled)


if __name__ == "__main__":
    unittest.main()
