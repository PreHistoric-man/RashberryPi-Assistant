"""Tests for the Developer Mode text-based AI test console."""

import threading
import time
import unittest
from unittest.mock import Mock

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.assistant import AssistantCore
from core.state_manager import StateManager
from ui.developer.ai_test_console import AITestConsole
from ui.developer.developer_controller import DeveloperController
from ui.states import AssistantState


class FakeMicrophone(QObject):
    audio_captured = Signal(bytes)
    error_occurred = Signal(str)

    def start_recording(self):
        pass

    def stop_recording(self):
        pass


class FakeTTS(QObject):
    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.calls = []

    def speak(self, text):
        self.calls.append(text)
        self.speech_started.emit()
        self.speech_finished.emit()

    def stop(self):
        pass


class BlockingFakeTTS(FakeTTS):
    def __init__(self):
        super().__init__()
        self.release_playback = threading.Event()

    def speak(self, text):
        self.calls.append(text)
        self.speech_started.emit()
        self.release_playback.wait(timeout=2)
        self.speech_finished.emit()


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
                "engine_status": "Local TinyLlama via llama.cpp",
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
        self.assertIn("Status: Local TinyLlama via llama.cpp", metrics)
        self.assertTrue(console.send_button.isEnabled())
        self.assertEqual(console.send_button.text(), "Generate")

    def test_generate_and_speak_uses_same_generation_and_speech_callbacks(self):
        generator = Mock(return_value=("Generated response.", {"model_name": "TinyLlama"}))
        speaker = Mock(return_value=True)
        console = AITestConsole(generator, speak_response=speaker)
        console.prompt_input.setPlainText("Say hello")

        self.assertTrue(console.submit_prompt(speak_after_generation=True))
        self.wait_for_generation(console)

        generator.assert_called_once_with("Say hello")
        speaker.assert_called_once_with("Generated response.")
        self.assertEqual(console.response_output.toPlainText(), "Generated response.")
        self.assertEqual(console.generation_status.text(), "Speaking...")
        self.assertFalse(console.generate_speak_button.isEnabled())
        self.assertFalse(console.speak_response_button.isEnabled())

    def test_generate_does_not_speak(self):
        generator = Mock(return_value=("Generated response.", {"model_name": "TinyLlama"}))
        speaker = Mock()
        console = AITestConsole(generator, speak_response=speaker)
        console.prompt_input.setPlainText("Generate only")

        self.assertTrue(console.submit_prompt())
        self.wait_for_generation(console)

        generator.assert_called_once_with("Generate only")
        speaker.assert_not_called()
        self.assertEqual(console.response_output.toPlainText(), "Generated response.")

    def test_speak_response_uses_displayed_text_without_generating(self):
        generator = Mock()
        speaker = Mock(return_value=True)
        console = AITestConsole(generator, speak_response=speaker)
        console.response_output.setPlainText("Existing response.")

        self.assertTrue(console.speak_current_response())

        generator.assert_not_called()
        speaker.assert_called_once_with("Existing response.")
        self.assertEqual(console.response_output.toPlainText(), "Existing response.")

    def test_speak_response_without_text_shows_message(self):
        generator = Mock()
        speaker = Mock()
        console = AITestConsole(generator, speak_response=speaker)

        self.assertFalse(console.speak_current_response())

        generator.assert_not_called()
        speaker.assert_not_called()
        self.assertIn("no response", console.generation_status.text().lower())

    def test_speaking_state_disables_actions_until_idle(self):
        console = AITestConsole(Mock(), speak_response=Mock(return_value=True))

        console.set_assistant_state(AssistantState.SPEAKING)
        self.assertFalse(console.send_button.isEnabled())
        self.assertFalse(console.generate_speak_button.isEnabled())
        self.assertFalse(console.speak_response_button.isEnabled())

        console.set_assistant_state(AssistantState.IDLE)
        self.assertTrue(console.send_button.isEnabled())
        self.assertTrue(console.generate_speak_button.isEnabled())
        self.assertTrue(console.speak_response_button.isEnabled())

    def test_tts_metrics_display_only_values_provided_by_engine(self):
        console = AITestConsole(
            Mock(),
            get_tts_metrics=lambda: {
                "voice": "en_US-lessac-medium",
                "synthesis_time_seconds": 0.09,
                "audio_duration_seconds": 1.84,
                "real_time_factor": 0.049,
            },
        )

        console.update_tts_metrics()

        metrics = console.tts_metrics_label.text()
        self.assertIn("en_US-lessac-medium", metrics)
        self.assertIn("Synthesis: 0.090s", metrics)
        self.assertIn("Audio: 1.840s", metrics)
        self.assertIn("RTF: 0.049", metrics)

    def test_console_uses_assistant_core_instances_for_generate_and_speak(self):
        response_engine = Mock()
        response_engine.generate_response.return_value = "Shared engine response."
        response_engine.last_generation_metrics = {
            "model_name": "Fake LLM",
            "generation_time_seconds": 0.1,
        }
        tts = FakeTTS()
        assistant = AssistantCore(
            state_manager=StateManager(),
            mic_recorder=FakeMicrophone(),
            stt_engine=object(),
            response_engine=response_engine,
            tts_engine=tts,
        )
        console = AITestConsole(
            assistant.generate_developer_response,
            speak_response=assistant.start_speaking,
            get_tts_metrics=assistant.get_tts_metrics,
        )
        assistant.state_manager.state_changed.connect(console.set_assistant_state)
        console.prompt_input.setPlainText("Use shared instances")

        self.assertTrue(console.submit_prompt(speak_after_generation=True))
        self.wait_for_generation(console)
        deadline = time.monotonic() + 2
        while assistant.state_manager.current_state != AssistantState.IDLE and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()

        self.assertIs(assistant.ai, response_engine)
        self.assertIs(assistant.tts, tts)
        response_engine.generate_response.assert_called_once_with("Use shared instances")
        self.assertEqual(tts.calls, ["Shared engine response."])
        self.assertEqual(
            console.response_output.toPlainText(),
            "Shared engine response.",
        )
        self.assertEqual(assistant.state_manager.current_state, AssistantState.IDLE)
        assistant._idle_return_timer.stop()
        assistant._error_recovery_timer.stop()

    def test_assistant_core_rejects_overlapping_speech(self):
        tts = BlockingFakeTTS()
        assistant = AssistantCore(
            state_manager=StateManager(),
            mic_recorder=FakeMicrophone(),
            stt_engine=object(),
            response_engine=Mock(),
            tts_engine=tts,
        )

        self.assertTrue(assistant.start_speaking("first"))
        self.assertFalse(assistant.start_speaking("second"))
        self.assertEqual(tts.calls, ["first"])

        tts.release_playback.set()
        deadline = time.monotonic() + 2
        while assistant._tts_lock.locked() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()

        self.assertFalse(assistant._tts_lock.locked())
        self.assertEqual(assistant.state_manager.current_state, AssistantState.IDLE)
        assistant._idle_return_timer.stop()
        assistant._error_recovery_timer.stop()

    def test_fallback_metrics_are_clearly_labeled(self):
        generator = Mock(return_value=(
            "Hello! How can I help you today?",
            {
                "model_name": "Development fallback",
                "generation_time_seconds": 0.0,
                "generated_tokens": 0,
                "tokens_per_second": 0.0,
                "engine_status": "Rule-based fallback; no LLM used (PI_ASSISTANT_LLM_MODEL_PATH is not set)",
            },
        ))
        console = AITestConsole(generator)
        console.prompt_input.setPlainText("Hello")

        self.assertTrue(console.submit_prompt())
        self.wait_for_generation(console)

        self.assertIn("Model: Development fallback", console.metrics_label.text())
        self.assertIn("Rule-based fallback; no LLM used", console.metrics_label.text())

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
