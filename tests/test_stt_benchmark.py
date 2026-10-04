"""Tests for STT configuration and standalone benchmark calculations."""

import os
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

from tools.benchmark_stt import (
    _build_configurations,
    build_parser,
    calculate_real_time_factor,
    calculate_word_error_rate,
    run_benchmark,
)
from core.assistant import AssistantCore
from core.state_manager import StateManager
from voice.microphone import MicrophoneRecorder
from voice.stt import faster_whisper_stt
from voice.stt.faster_whisper_stt import FasterWhisperSTT


class FakeBenchmarkModel:
    def __init__(self):
        self.transcription_count = 0

    def transcribe(self, audio, **kwargs):
        self.transcription_count += 1
        return iter([SimpleNamespace(text="recognized words", tokens=[1, 2])]), None


class FakeTTS(QObject):
    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def stop(self):
        pass

    def speak(self, text):
        pass


class BlockingSTT:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_id = None

    def transcribe(self, audio):
        self.thread_id = threading.get_ident()
        self.started.set()
        self.release.wait(timeout=2)
        return "recognized"


class STTConfigurationTests(unittest.TestCase):
    def test_environment_config_selects_model_and_threads_and_reuses_loaded_model(self):
        model = FakeBenchmarkModel()
        with (
            patch.dict(os.environ, {
                "PI_ASSISTANT_STT_MODEL": "tiny.en",
                "PI_ASSISTANT_STT_CPU_THREADS": "3",
            }),
            patch.object(faster_whisper_stt, "FASTER_WHISPER_AVAILABLE", True),
            patch.object(faster_whisper_stt, "WhisperModel", return_value=model) as factory,
        ):
            engine = FasterWhisperSTT()
            self.assertEqual(engine.model_size, "tiny.en")
            self.assertEqual(engine.cpu_threads, 3)
            self.assertIn("tiny.en", engine.engine_name)
            self.assertIn("3 threads", engine.engine_name)

            engine.transcribe(np.zeros(1600, dtype=np.float32))
            engine.transcribe(np.zeros(1600, dtype=np.float32))

        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["cpu_threads"], 3)
        self.assertEqual(model.transcription_count, 2)

    def test_explicit_model_configuration_overrides_environment(self):
        with (
            patch.dict(os.environ, {
                "PI_ASSISTANT_STT_MODEL": "tiny.en",
                "PI_ASSISTANT_STT_CPU_THREADS": "3",
            }),
            patch.object(faster_whisper_stt, "FASTER_WHISPER_AVAILABLE", True),
            patch.object(faster_whisper_stt, "WhisperModel", return_value=FakeBenchmarkModel()),
        ):
            engine = FasterWhisperSTT(model_size="base.en", cpu_threads=2)

        self.assertEqual(engine.model_size, "base.en")
        self.assertEqual(engine.cpu_threads, 2)
        self.assertIn("base.en", engine.engine_name)
        self.assertIn("2 threads", engine.engine_name)

    def test_default_thread_count_reserves_ui_headroom(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(FasterWhisperSTT._resolve_cpu_threads(None), 3)


class BenchmarkMetricTests(unittest.TestCase):
    def test_benchmark_modes_select_requested_configurations(self):
        parser = build_parser()
        model_args = parser.parse_args(["--audio", "clips", "--mode", "models"])
        thread_args = parser.parse_args(["--audio", "clips", "--mode", "threads"])
        repeat_args = parser.parse_args(["--audio", "clips", "--mode", "repeat", "--model", "base.en", "--threads", "3"])

        self.assertEqual(_build_configurations(model_args), [("tiny.en", 4), ("base.en", 4), ("small", 4)])
        self.assertEqual(_build_configurations(thread_args), [("tiny.en", 2), ("tiny.en", 3), ("tiny.en", 4)])
        self.assertEqual(_build_configurations(repeat_args), [("base.en", 3)])

    def test_rtf_and_word_error_rate_calculations(self):
        self.assertEqual(calculate_real_time_factor(1.0, 4.0), 0.25)
        self.assertIsNone(calculate_real_time_factor(1.0, 0.0))
        self.assertEqual(calculate_word_error_rate("Hello, assistant", "hello friend"), 0.5)
        self.assertIsNone(calculate_word_error_rate("", "recognized words"))

    def test_valid_wav_runs_warmup_then_repeated_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, np.zeros(16000, dtype=np.float32), 16000)
            model = FakeBenchmarkModel()
            model_factory = Mock(return_value=model)

            report = run_benchmark(
                [wav_path],
                [("tiny.en", 2)],
                runs=2,
                references={"sample.wav": "recognized words"},
                model_factory=model_factory,
            )

        model_factory.assert_called_once()
        self.assertEqual(model.transcription_count, 3)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][0]["audio_duration_seconds"], 1.0)
        self.assertEqual(report["results"][0]["segment_count"], 1)
        self.assertEqual(report["results"][0]["token_count"], 2)
        self.assertEqual(report["results"][0]["word_error_rate"], 0.0)
        self.assertEqual(report["errors"], [])

    def test_empty_wav_is_reported_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_wav = Path(temp_dir) / "empty.wav"
            empty_wav.write_bytes(b"")
            model_factory = Mock()

            report = run_benchmark(
                [empty_wav],
                [("tiny.en", 2)],
                model_factory=model_factory,
            )

        model_factory.assert_not_called()
        self.assertEqual(report["results"], [])
        self.assertIn("Could not read WAV", report["errors"][0]["error"])

    def test_blocking_transcription_does_not_block_qt_event_loop(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        stt = BlockingSTT()
        assistant = AssistantCore(
            state_manager=StateManager(),
            mic_recorder=MicrophoneRecorder(),
            stt_engine=stt,
            response_engine=object(),
            tts_engine=FakeTTS(),
        )
        transcription_done = threading.Event()
        assistant.transcription_ready.connect(lambda _text: transcription_done.set())
        timer_fired = threading.Event()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: timer_fired.set())

        try:
            main_thread_id = threading.get_ident()
            assistant._on_audio_captured(b"fake wav")
            self.assertTrue(stt.started.wait(timeout=1))
            timer.start(1)

            deadline = time.monotonic() + 1
            while not timer_fired.is_set() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.001)

            self.assertTrue(timer_fired.is_set())
            self.assertNotEqual(stt.thread_id, main_thread_id)

            stt.release.set()
            deadline = time.monotonic() + 1
            while not transcription_done.is_set() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.001)
            self.assertTrue(transcription_done.is_set())
        finally:
            stt.release.set()
            assistant._idle_return_timer.stop()
            assistant.deleteLater()


if __name__ == "__main__":
    unittest.main()