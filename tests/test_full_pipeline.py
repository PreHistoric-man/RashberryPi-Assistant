"""Deterministic end-to-end and optional real-model voice pipeline checks."""

import os
from pathlib import Path
import threading
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from ai.response_engine import LocalResponseEngine
from core.assistant import AssistantCore
from core.state_manager import StateManager
from ui.face_widget import FaceWidget
from ui.states import AssistantState
from voice.text_to_speech import SpeechAudio
from voice.tts.piper_tts import PiperTextToSpeechEngine


REAL_TTS_MODEL = os.getenv("PI_ASSISTANT_TTS_MODEL_PATH", "").strip()
REAL_LLM_MODEL = os.getenv("PI_ASSISTANT_LLM_MODEL_PATH", "").strip()


def model_available(path):
    return bool(path) and Path(path).is_file()


class FakeAudioPlayer:
    def __init__(self, events=None, block=False):
        self.events = events if events is not None else []
        self.received_audio = []
        self.entered_playback = threading.Event()
        self.finish_playback = threading.Event()
        if not block:
            self.finish_playback.set()

    def play(self, audio):
        self.events.append(("audio_player", threading.get_ident()))
        self.received_audio.append(audio)
        self.entered_playback.set()
        self.finish_playback.wait(timeout=3)

    def stop(self):
        self.finish_playback.set()


class FakeMicrophone(QObject):
    audio_captured = Signal(bytes)
    error_occurred = Signal(str)

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.last_duration = 0.25
        self.silence_tail_duration = 0.05
        self.time_until_speech_detected = 0.1
        self.total_capture_time = 0.25
        self.silence_duration = 0.5

    def start_recording(self):
        self.events.append(("microphone_capture", threading.get_ident()))
        self.audio_captured.emit(b"fake wav")

    def stop_recording(self):
        pass


class FakeSTT:
    engine_name = "FakeSTT"

    def __init__(self, events, results=None):
        self.events = events
        self.results = list(results or ["What is Python?"])
        self.calls = 0
        self.thread_ids = []

    def transcribe(self, audio):
        self.events.append(("stt", threading.get_ident()))
        self.thread_ids.append(threading.get_ident())
        self.calls += 1
        outcome = self.results.pop(0) if self.results else "What is Python?"
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeLLM:
    def __init__(self, events, results=None, gate=None):
        self.events = events
        self.results = list(results or ["Python is a programming language."])
        self.gate = gate
        self.calls = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.thread_ids = []
        self._lock = threading.Lock()
        self.last_generation_metrics = {
            "model_name": "FakeLLM",
            "generation_time_seconds": 0.01,
            "generated_tokens": 6,
            "tokens_per_second": 600.0,
        }

    def generate_response(self, prompt):
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            self.events.append(("llm", threading.get_ident()))
            self.thread_ids.append(threading.get_ident())
            self.calls.append(prompt)
            if self.gate is not None:
                self.gate.wait(timeout=3)
            outcome = self.results.pop(0) if self.results else "Python is a programming language."
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            with self._lock:
                self.active_calls -= 1


class FakeTTS(QObject):
    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, events, player, results=None):
        super().__init__()
        self.events = events
        self.player = player
        self.results = list(results or [None])
        self.calls = []
        self.thread_ids = []

    def speak(self, text):
        self.events.append(("tts", threading.get_ident()))
        self.thread_ids.append(threading.get_ident())
        self.calls.append(text)
        outcome = self.results.pop(0) if self.results else None
        self.speech_started.emit()
        try:
            if isinstance(outcome, BaseException):
                raise outcome
            audio = SpeechAudio(samples=np.zeros(16, dtype="int16"), sample_rate=16000)
            self.player.play(audio)
        except Exception as exc:
            self.error_occurred.emit(f"TTS failure: {exc}")
            raise
        else:
            self.speech_finished.emit()

    def stop(self):
        self.player.stop()


class FullPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.ui_thread_id = threading.get_ident()

    def setUp(self):
        self.events = []
        self.player = FakeAudioPlayer(self.events, block=True)
        self.tts = FakeTTS(self.events, self.player)
        self.stt = FakeSTT(self.events)
        self.llm = FakeLLM(self.events)
        self.state_manager = StateManager()
        self.mic = FakeMicrophone(self.events)
        self.face = FaceWidget()
        self.state_manager.state_changed.connect(self.face.set_state)
        self.states = []
        self.state_manager.state_changed.connect(self.states.append)
        self.assistant = AssistantCore(
            state_manager=self.state_manager,
            mic_recorder=self.mic,
            stt_engine=self.stt,
            response_engine=self.llm,
            tts_engine=self.tts,
        )
        self.assistant._idle_return_timer.setInterval(35)
        self.assistant._error_recovery_timer.setInterval(35)
        self.timer_instances = (
            self.face._speech_timer,
            self.face._viseme_change_timer,
            self.face._thinking_timer,
            self.face._listening_timer,
            self.face._blink_timer,
            self.face._blink_anim_timer,
        )

    def tearDown(self):
        self.player.finish_playback.set()
        self.assistant._idle_return_timer.stop()
        self.assistant._error_recovery_timer.stop()
        for timer in self.timer_instances:
            timer.stop()
        self.face.close()
        self.assistant.deleteLater()

    def wait_until(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.002)
        self.app.processEvents()
        return predicate()

    def run_interaction_until_playback(self, expected_player_calls):
        self.player.entered_playback.clear()
        self.player.finish_playback.clear()
        self.assistant.start_interaction()
        self.assertTrue(
            self.wait_until(
                lambda: self.player.entered_playback.is_set()
                and self.state_manager.current_state == AssistantState.SPEAKING
            )
        )
        self.assertEqual(len(self.player.received_audio), expected_player_calls)
        self.assertEqual(self.state_manager.current_state, AssistantState.SPEAKING)

    def finish_playback(self):
        self.player.finish_playback.set()
        self.assertTrue(
            self.wait_until(
                lambda: self.state_manager.current_state == AssistantState.IDLE
            )
        )

    def test_full_pipeline_order_states_face_and_three_reused_interactions(self):
        state_sequences = []
        timer_ids = tuple(id(timer) for timer in self.timer_instances)

        for index in range(3):
            state_start = len(self.states)
            expected_call_count = index + 1
            self.run_interaction_until_playback(expected_call_count)

            observed_before_playback_end = self.states[state_start:]
            self.assertEqual(
                observed_before_playback_end,
                [
                    AssistantState.LISTENING,
                    AssistantState.THINKING,
                    AssistantState.SPEAKING,
                ],
            )
            self.assertTrue(self.face._speech_timer.isActive())
            self.assertTrue(self.face._viseme_change_timer.isActive())
            self.assertFalse(self.face._thinking_timer.isActive())
            self.assertFalse(self.face._listening_timer.isActive())
            self.assertEqual(self.assistant.last_response, "Python is a programming language.")
            self.finish_playback()
            state_sequences.append(self.states[state_start:])

            self.assertEqual(self.face.get_state(), AssistantState.IDLE)
            self.assertFalse(self.face._speech_timer.isActive())
            self.assertFalse(self.face._viseme_change_timer.isActive())
            self.assertTrue(self.face._blink_timer.isActive())
            self.assertEqual(tuple(id(timer) for timer in self.timer_instances), timer_ids)

        expected_sequence = [
            AssistantState.LISTENING,
            AssistantState.THINKING,
            AssistantState.SPEAKING,
            AssistantState.IDLE,
        ]
        self.assertEqual(state_sequences, [expected_sequence] * 3)
        operation_names = [event[0] for event in self.events]
        for offset in range(0, len(operation_names), 5):
            self.assertEqual(
                operation_names[offset:offset + 5],
                ["microphone_capture", "stt", "llm", "tts", "audio_player"],
            )
        self.assertEqual(self.stt.calls, 3)
        self.assertEqual(self.llm.calls, ["What is Python?"] * 3)
        self.assertEqual(
            self.tts.calls,
            ["Python is a programming language."] * 3,
        )
        self.assertEqual(len(self.player.received_audio), 3)
        self.assertIs(self.assistant.ai, self.llm)
        self.assertIs(self.assistant.tts, self.tts)
        self.assertTrue(all(tid != self.ui_thread_id for tid in self.stt.thread_ids))
        self.assertTrue(all(tid != self.ui_thread_id for tid in self.llm.thread_ids))
        self.assertTrue(all(tid != self.ui_thread_id for tid in self.tts.thread_ids))
        worker_events = {
            "stt",
            "llm",
            "tts",
            "audio_player",
        }
        self.assertTrue(
            all(
                thread_id != self.ui_thread_id
                for name, thread_id in self.events
                if name in worker_events
            )
        )

    def test_slow_generation_keeps_qt_event_loop_responsive(self):
        gate = threading.Event()
        self.llm.gate = gate
        ticked = []
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: ticked.append(True))

        self.assistant.start_interaction()
        self.assertTrue(self.wait_until(lambda: self.stt.calls == 1))
        self.assertTrue(self.wait_until(lambda: bool(self.llm.calls)))
        timer.start(0)
        self.app.processEvents()
        self.assertTrue(ticked)
        self.assertEqual(self.state_manager.current_state, AssistantState.THINKING)
        self.assertEqual(self.tts.calls, [])
        gate.set()
        self.assertTrue(self.wait_until(lambda: self.player.entered_playback.is_set()))
        self.finish_playback()

    def test_overlapping_generation_is_serialized_by_existing_lock(self):
        gate = threading.Event()
        self.llm.gate = gate
        threads = [
            threading.Thread(
                target=self.assistant.generate_developer_response,
                args=(f"prompt {index}",),
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(self.wait_until(lambda: self.llm.active_calls == 1))
        time.sleep(0.02)
        self.assertEqual(self.llm.max_active_calls, 1)
        gate.set()
        for thread in threads:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(self.llm.max_active_calls, 1)

    def test_overlapping_speech_is_rejected_until_finished(self):
        self.assertTrue(self.assistant.start_speaking("first response"))
        self.assertTrue(self.wait_until(lambda: self.player.entered_playback.is_set()))
        self.assertFalse(self.assistant.start_speaking("second response"))
        self.assertEqual(self.tts.calls, ["first response"])
        self.assertEqual(len(self.player.received_audio), 1)
        self.assertEqual(self.state_manager.current_state, AssistantState.SPEAKING)
        self.finish_playback()
        self.player.entered_playback.clear()
        self.player.finish_playback.clear()
        self.assertTrue(self.assistant.start_speaking("second response"))
        self.assertTrue(self.wait_until(lambda: self.player.entered_playback.is_set()))
        self.assertEqual(len(self.tts.calls), 2)
        self.finish_playback()

    def test_stt_failure_recovers_and_later_interaction_succeeds(self):
        self.stt.results = [RuntimeError("fake STT failure"), "What is Python?"]
        self.assistant.start_interaction()
        self.assertTrue(
            self.wait_until(
                lambda: self.state_manager.current_state == AssistantState.IDLE
            )
        )
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.tts.calls, [])

        self.run_interaction_until_playback(1)
        self.finish_playback()
        self.assertEqual(self.stt.calls, 2)
        self.assertEqual(self.llm.calls, ["What is Python?"])

    def test_llm_failure_recovers_without_speech_and_later_succeeds(self):
        self.llm.results = [RuntimeError("fake LLM failure"), "Recovered response."]
        self.assistant.start_interaction()
        self.assertTrue(
            self.wait_until(
                lambda: self.state_manager.current_state == AssistantState.IDLE
            )
        )
        self.assertEqual(self.tts.calls, [])

        self.run_interaction_until_playback(1)
        self.assertEqual(self.tts.calls, ["Recovered response."])
        self.finish_playback()

    def test_empty_llm_response_does_not_start_tts_and_recovers(self):
        self.llm.results = ["   ", "Recovered response."]
        self.assistant.start_interaction()
        self.assertTrue(
            self.wait_until(
                lambda: self.state_manager.current_state == AssistantState.IDLE
            )
        )
        self.assertEqual(self.tts.calls, [])

        self.run_interaction_until_playback(1)
        self.finish_playback()
        self.assertEqual(self.tts.calls, ["Recovered response."])

    def test_latency_budget_uses_measured_stages_and_handoff_only(self):
        self.assistant._post_capture_start_time = time.perf_counter() - 1.0
        self.assistant._stt_processing_time = 0.2
        self.assistant._llm_processing_time = 0.3
        self.assistant._llm_generation_metrics = {"llm_wall_time_seconds": 0.3}
        self.assistant._silence_tail_duration = 0.5
        self.assistant._tts_worker_start_time = time.perf_counter() - 0.15
        self.tts.last_synthesis_metrics = {
            "synthesis_time_seconds": 0.1,
            "playback_handoff_seconds": 0.12,
        }

        self.assistant._on_tts_finished()

        metrics = self.assistant.get_latency_metrics()
        self.assertEqual(metrics["stt_seconds"], 0.2)
        self.assertEqual(metrics["llm_seconds"], 0.3)
        self.assertEqual(metrics["tts_seconds"], 0.1)
        self.assertIsNotNone(metrics["total_seconds"])
        self.assertGreater(metrics["total_seconds"], 0.5)
        self.assertEqual(metrics["target_seconds"], 5.0)

        self.tts.last_synthesis_metrics = {"synthesis_time_seconds": 0.2}
        self.assistant._on_tts_finished()
        self.assertIsNone(self.assistant.get_latency_metrics()["total_seconds"])

    def test_tts_failure_reports_error_recovers_and_later_succeeds(self):
        self.tts.results = [RuntimeError("fake TTS failure"), None]
        errors = []
        self.assistant.error_occurred.connect(errors.append)
        self.assistant.start_interaction()
        self.assertTrue(
            self.wait_until(
                lambda: self.state_manager.current_state == AssistantState.IDLE
            )
        )
        self.assertTrue(any("fake TTS failure" in error for error in errors))
        self.assertFalse(self.assistant._tts_lock.locked())
        self.assertFalse(self.face._speech_timer.isActive())
        self.assertEqual(len(self.player.received_audio), 0)

        self.run_interaction_until_playback(1)
        self.finish_playback()
        self.assertEqual(len(self.player.received_audio), 1)


@unittest.skipUnless(
    model_available(REAL_TTS_MODEL),
    "PI_ASSISTANT_TTS_MODEL_PATH is not configured to an existing Piper model",
)
class RealPiperPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_assistant_core_reuses_configured_piper_across_repeated_speech(self):
        player = FakeAudioPlayer(block=True)
        tts = PiperTextToSpeechEngine(audio_player=player)
        manager = StateManager()
        assistant = AssistantCore(
            state_manager=manager,
            mic_recorder=FakeMicrophone([]),
            stt_engine=FakeSTT([]),
            response_engine=FakeLLM([]),
            tts_engine=tts,
        )
        assistant._error_recovery_timer.setInterval(50)
        try:
            states = []
            manager.state_changed.connect(states.append)
            self.assertEqual(tts.model_load_count, 1)
            for index in range(2):
                player.entered_playback.clear()
                player.finish_playback.clear()
                self.assertTrue(assistant.start_speaking(f"Real Piper test {index}."))
                self.assertTrue(self._wait(player.entered_playback.is_set))
                self.app.processEvents()
                self.assertEqual(manager.current_state, AssistantState.SPEAKING)
                player.finish_playback.set()
                self.assertTrue(
                    self._wait(
                        lambda: manager.current_state == AssistantState.IDLE
                    )
                )
                self.assertEqual(tts.model_load_count, 1)
            self.assertEqual(len(player.received_audio), 2)
            metrics = tts.last_synthesis_metrics
            self.assertEqual(metrics["voice"], Path(REAL_TTS_MODEL).stem)
            self.assertEqual(metrics["model_name"], Path(REAL_TTS_MODEL).stem)
            self.assertGreater(metrics["synthesis_time_seconds"], 0)
            self.assertGreater(metrics["audio_duration_seconds"], 0)
            self.assertGreater(metrics["real_time_factor"], 0)
            self.assertGreaterEqual(
                metrics["playback_handoff_seconds"],
                metrics["synthesis_time_seconds"],
            )
            self.assertEqual(
                states,
                [
                    AssistantState.SPEAKING,
                    AssistantState.IDLE,
                    AssistantState.SPEAKING,
                    AssistantState.IDLE,
                ],
            )
        finally:
            player.finish_playback.set()
            assistant._idle_return_timer.stop()
            assistant._error_recovery_timer.stop()
            assistant.deleteLater()

    def _wait(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.002)
        self.app.processEvents()
        return predicate()


@unittest.skipUnless(
    model_available(REAL_LLM_MODEL) and model_available(REAL_TTS_MODEL),
    "real TinyLlama and Piper models are not both configured",
)
class RealModelsPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_real_tinyllama_piper_audio_and_state_path(self):
        player = FakeAudioPlayer(block=True)
        tts = PiperTextToSpeechEngine(audio_player=player)
        response_engine = LocalResponseEngine()
        llm_instance = response_engine.llm
        manager = StateManager()
        assistant = AssistantCore(
            state_manager=manager,
            mic_recorder=FakeMicrophone([]),
            stt_engine=FakeSTT([]),
            response_engine=response_engine,
            tts_engine=tts,
        )
        assistant._error_recovery_timer.setInterval(50)
        try:
            states = []
            manager.state_changed.connect(states.append)
            worker = threading.Thread(
                target=assistant._generate_response_worker,
                args=("Say one short sentence about Python.",),
                daemon=True,
            )
            worker.start()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and worker.is_alive():
                self.app.processEvents()
                time.sleep(0.01)
            self.assertFalse(worker.is_alive(), "TinyLlama generation did not finish in 120 seconds")
            self.assertTrue(assistant.last_response.strip())
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not player.entered_playback.is_set():
                self.app.processEvents()
                time.sleep(0.01)
            self.assertTrue(player.entered_playback.is_set(), "real pipeline did not reach AudioPlayer")
            self.app.processEvents()
            self.assertEqual(manager.current_state, AssistantState.SPEAKING)
            self.assertEqual(len(player.received_audio), 1)
            self.assertGreater(player.received_audio[0].samples.size, 0)
            self.assertIs(assistant.ai, response_engine)
            self.assertIs(response_engine.llm, llm_instance)
            llm_metrics = response_engine.last_generation_metrics
            self.assertGreater(llm_metrics["generation_time_seconds"], 0)
            self.assertGreater(llm_metrics["generated_tokens"], 0)
            self.assertGreater(llm_metrics["tokens_per_second"], 0)
            self.assertEqual(tts.model_load_count, 1)
            self.assertEqual(states, [AssistantState.SPEAKING])
            player.finish_playback.set()
            self.assertTrue(
                self._wait(lambda: manager.current_state == AssistantState.IDLE)
            )
            self.assertEqual(states, [AssistantState.SPEAKING, AssistantState.IDLE])
        finally:
            player.finish_playback.set()
            assistant._idle_return_timer.stop()
            assistant._error_recovery_timer.stop()
            assistant.deleteLater()

    def _wait(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.002)
        self.app.processEvents()
        return predicate()


if __name__ == "__main__":
    unittest.main()
