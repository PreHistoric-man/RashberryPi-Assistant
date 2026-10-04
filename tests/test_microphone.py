"""Focused tests for microphone capture and energy-based end-of-speech detection."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal

from core.assistant import AssistantCore
from core.state_manager import StateManager
from ui.states import AssistantState
from voice.microphone import MicrophoneRecorder


class FakeInputStream:
    def __init__(self, blocks):
        self.blocks = iter(blocks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, frames):
        try:
            block = next(self.blocks)
        except StopIteration:
            block = np.zeros(frames, dtype=np.float32)
        return block[:frames], False


class BlockingInputStream(FakeInputStream):
    def __init__(self):
        self.read_entered = threading.Event()
        self.release_read = threading.Event()

    def read(self, frames):
        self.read_entered.set()
        self.release_read.wait(timeout=2)
        return np.full(frames, 0.03, dtype=np.float32), False


class FakeTTS(QObject):
    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def stop(self):
        pass

    def speak(self, text):
        pass


class FakeSTT:
    def transcribe(self, audio):
        return "recognized"


class MicrophoneRecorderTests(unittest.TestCase):
    sample_rate = 1000
    block_samples = 100

    def setUp(self):
        self.silent = np.zeros(self.block_samples, dtype=np.float32)
        self.voice = np.full(self.block_samples, 0.03, dtype=np.float32)

    def test_default_configuration_matches_phase_one_values(self):
        recorder = MicrophoneRecorder()

        self.assertEqual(recorder.sample_rate, 16000)
        self.assertEqual(recorder.silence_threshold_rms, 0.015)
        self.assertEqual(recorder.silence_duration_seconds, 0.5)
        self.assertEqual(recorder.max_recording_duration_seconds, 8.0)
        self.assertEqual(recorder.energy_window_ms, 100)
        self.assertEqual(MicrophoneRecorder(energy_window_ms=50).energy_window_ms, 50)
        self.assertEqual(MicrophoneRecorder(energy_window_ms=500).energy_window_ms, 100)

    def capture(self, blocks, *, silence_duration=0.5, max_duration=2.0):
        recorder = MicrophoneRecorder(
            sample_rate=self.sample_rate,
            silence_threshold_rms=0.015,
            silence_duration_seconds=silence_duration,
            max_recording_duration_seconds=max_duration,
            energy_window_ms=100,
        )
        recorder.is_available = lambda: True
        with patch("voice.microphone.sd.InputStream", return_value=FakeInputStream(blocks)):
            recorder._record_worker()
        return recorder

    def test_initial_silence_does_not_end_capture_before_speech(self):
        recorder = self.capture(
            [self.silent, self.silent, self.silent, self.voice, self.voice]
            + [self.silent] * 5
        )

        self.assertTrue(recorder.speech_detected_in_last_recording)
        self.assertAlmostEqual(recorder.last_duration, 1.0, places=2)
        self.assertAlmostEqual(recorder.silence_tail_duration, 0.5, places=2)

    def test_continuous_speech_stops_at_maximum_duration(self):
        recorder = self.capture([self.voice] * 10, max_duration=0.55)

        self.assertTrue(recorder.speech_detected_in_last_recording)
        self.assertAlmostEqual(recorder.last_duration, 0.55, places=2)
        self.assertEqual(recorder.silence_tail_duration, 0.0)

    def test_short_pause_does_not_end_capture(self):
        recorder = self.capture(
            [self.voice, self.voice, self.silent, self.silent, self.voice, self.voice]
            + [self.silent] * 5
        )

        self.assertAlmostEqual(recorder.last_duration, 1.1, places=2)
        self.assertAlmostEqual(recorder.silence_tail_duration, 0.5, places=2)

    def test_non_window_aligned_silence_threshold_is_not_floored(self):
        recorder = self.capture(
            [self.voice] + [self.silent] * 6,
            silence_duration=0.55,
        )

        self.assertAlmostEqual(recorder.silence_tail_duration, 0.6, places=2)
        self.assertAlmostEqual(recorder.last_duration, 0.7, places=2)

    def test_completely_silent_input_waits_until_maximum_duration(self):
        recorder = self.capture([], max_duration=0.35)

        self.assertFalse(recorder.speech_detected_in_last_recording)
        self.assertAlmostEqual(recorder.last_duration, 0.35, places=2)
        self.assertEqual(recorder.silence_tail_duration, 0.0)

    def test_start_recording_runs_capture_in_background(self):
        recorder = MicrophoneRecorder(
            sample_rate=self.sample_rate,
            max_recording_duration_seconds=0.1,
            energy_window_ms=100,
        )
        stream = BlockingInputStream()
        with patch.object(recorder, "is_available", return_value=True):
            with patch("voice.microphone.sd.InputStream", return_value=stream):
                recorder.start_recording()
                self.assertTrue(stream.read_entered.wait(timeout=1))
                self.assertTrue(recorder.is_recording)
                stream.release_read.set()
                recorder._record_thread.join(timeout=2)

        self.assertFalse(recorder._record_thread.is_alive())
        self.assertFalse(recorder.is_recording)

    def test_captured_audio_preserves_assistant_state_transitions(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        state_manager = StateManager()
        recorder = MicrophoneRecorder()
        assistant = AssistantCore(
            state_manager=state_manager,
            mic_recorder=recorder,
            stt_engine=FakeSTT(),
            response_engine=object(),
            tts_engine=FakeTTS(),
        )
        assistant._idle_return_timer.setInterval(1)
        transitions = []
        state_manager.state_changed.connect(lambda state: transitions.append(state.name))

        state_manager.set_state(AssistantState.LISTENING)
        assistant._on_audio_captured(b"fake wav")

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and state_manager.current_state.name != "IDLE":
            app.processEvents()
            time.sleep(0.001)
        assistant.deleteLater()

        self.assertEqual(transitions, ["LISTENING", "THINKING", "IDLE"])


if __name__ == "__main__":
    unittest.main()