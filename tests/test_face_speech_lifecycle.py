"""Verify face animation follows AssistantCore and TTS lifecycle states."""

import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal

from core.assistant import AssistantCore
from core.state_manager import StateManager
from ui.face_widget import FaceWidget
from ui.states import AssistantState


class FakeMicrophone(QObject):
    audio_captured = Signal(bytes)
    error_occurred = Signal(str)

    def start_recording(self):
        pass

    def stop_recording(self):
        pass


class FakeResponseEngine:
    def generate_response(self, text):
        return "A test response."


class FakeTTS(QObject):
    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.entered = threading.Event()
        self.begin_speech = threading.Event()
        self.finish_speech = threading.Event()

    def speak(self, text):
        self.entered.set()
        self.begin_speech.wait(timeout=2)
        self.speech_started.emit()
        self.finish_speech.wait(timeout=2)
        if self.fail:
            self.error_occurred.emit("simulated TTS failure")
            raise RuntimeError("simulated TTS failure")
        self.speech_finished.emit()

    def stop(self):
        pass


class FaceSpeechLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.state_manager = StateManager()
        self.face = FaceWidget()
        self.state_manager.state_changed.connect(self.face.set_state)
        self.tts = FakeTTS()
        self.assistant = AssistantCore(
            state_manager=self.state_manager,
            mic_recorder=FakeMicrophone(),
            stt_engine=object(),
            response_engine=FakeResponseEngine(),
            tts_engine=self.tts,
        )
        self.assistant._error_recovery_timer.setInterval(40)
        self.states = []
        self.state_manager.state_changed.connect(
            lambda state: self.states.append(state)
        )
        self.animation_timers = (
            self.face._speech_timer,
            self.face._viseme_change_timer,
            self.face._thinking_timer,
            self.face._listening_timer,
            self.face._blink_timer,
        )

    def tearDown(self):
        self.tts.begin_speech.set()
        self.tts.finish_speech.set()
        self.assistant._idle_return_timer.stop()
        self.assistant._error_recovery_timer.stop()
        for timer in self.animation_timers:
            timer.stop()
        self.face.close()
        self.assistant.deleteLater()

    def process_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.002)
        self.app.processEvents()
        return predicate()

    def begin_response(self):
        self.state_manager.set_state(AssistantState.LISTENING)
        self.state_manager.set_state(AssistantState.THINKING)
        self.assistant._on_worker_response_done("A test response.")
        self.assertTrue(self.tts.entered.wait(timeout=1))

    def test_speaking_animation_tracks_tts_start_and_finish(self):
        self.begin_response()

        self.assertEqual(self.face.get_state(), AssistantState.THINKING)
        self.assertTrue(self.face._thinking_timer.isActive())
        self.assertFalse(self.face._speech_timer.isActive())
        self.assertFalse(self.face._viseme_change_timer.isActive())

        self.tts.begin_speech.set()
        self.assertTrue(
            self.process_until(
                lambda: self.face.get_state() == AssistantState.SPEAKING
            )
        )
        self.assertTrue(self.face._speech_timer.isActive())
        self.assertTrue(self.face._viseme_change_timer.isActive())
        self.assertFalse(self.face._thinking_timer.isActive())

        self.tts.finish_speech.set()
        self.assertTrue(
            self.process_until(
                lambda: self.face.get_state() == AssistantState.IDLE
            )
        )
        self.assertEqual(
            self.states,
            [
                AssistantState.LISTENING,
                AssistantState.THINKING,
                AssistantState.SPEAKING,
                AssistantState.IDLE,
            ],
        )
        self.assertFalse(self.face._speech_timer.isActive())
        self.assertFalse(self.face._viseme_change_timer.isActive())
        self.assertTrue(self.face._blink_timer.isActive())
        self.assertEqual(
            self.animation_timers,
            (
                self.face._speech_timer,
                self.face._viseme_change_timer,
                self.face._thinking_timer,
                self.face._listening_timer,
                self.face._blink_timer,
            ),
        )

    def test_tts_error_stops_speaking_animation_and_returns_safely(self):
        self.tts.fail = True
        self.begin_response()
        self.tts.begin_speech.set()
        self.assertTrue(
            self.process_until(
                lambda: self.face.get_state() == AssistantState.SPEAKING
            )
        )
        self.tts.finish_speech.set()

        self.assertTrue(
            self.process_until(
                lambda: self.face.get_state() == AssistantState.IDLE
            )
        )
        self.assertIn(AssistantState.ERROR, self.states)
        self.assertFalse(self.face._speech_timer.isActive())
        self.assertFalse(self.face._viseme_change_timer.isActive())
        self.assertTrue(self.face._blink_timer.isActive())

    def test_non_speaking_states_keep_their_existing_animation_timers(self):
        self.state_manager.set_state(AssistantState.LISTENING)
        self.assertTrue(self.face._listening_timer.isActive())
        self.assertFalse(self.face._speech_timer.isActive())

        self.state_manager.set_state(AssistantState.THINKING)
        self.assertTrue(self.face._thinking_timer.isActive())
        self.assertFalse(self.face._listening_timer.isActive())
        self.assertFalse(self.face._speech_timer.isActive())

        self.state_manager.set_state(AssistantState.ERROR)
        self.assertFalse(self.face._thinking_timer.isActive())
        self.assertFalse(self.face._listening_timer.isActive())
        self.assertFalse(self.face._speech_timer.isActive())
        self.assertFalse(self.face._viseme_change_timer.isActive())


if __name__ == "__main__":
    unittest.main()
