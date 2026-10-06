"""Unit tests for the model-agnostic and Piper speech synthesis interfaces."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from voice.text_to_speech import BaseTextToSpeechEngine
from voice.tts.piper_tts import PiperTextToSpeechEngine


class FakePiperVoice:
    def __init__(self, sample_rate=22050):
        self.config = type("VoiceConfig", (), {"sample_rate": sample_rate})()
        self.synthesis_calls = []

    def synthesize_wav(self, text, wav_file):
        self.synthesis_calls.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.config.sample_rate)
        wav_file.writeframes(np.array([0, 100, -100, 0], dtype="<i2").tobytes())


class TextToSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "voice.onnx"
        self.model_path.write_bytes(b"fake model")

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_engine(self, **kwargs):
        voice = FakePiperVoice()
        loader = kwargs.pop("voice_loader", Mock(return_value=voice))
        player = kwargs.pop("audio_player", Mock())
        engine = PiperTextToSpeechEngine(
            self.model_path,
            voice_loader=loader,
            audio_player=player,
            **kwargs,
        )
        return engine, voice, loader, player

    def test_base_tts_interface_requires_all_operations(self):
        with self.assertRaises(TypeError):
            BaseTextToSpeechEngine()

        engine, _, _, _ = self.create_engine()
        self.assertIsInstance(engine, BaseTextToSpeechEngine)

    def test_missing_model_path_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PI_ASSISTANT_TTS_MODEL_PATH"):
                PiperTextToSpeechEngine()

    def test_invalid_model_path(self):
        missing_path = Path(self.temp_dir.name) / "missing.onnx"
        with self.assertRaisesRegex(FileNotFoundError, "Piper model file not found"):
            PiperTextToSpeechEngine(missing_path, voice_loader=Mock())

    def test_successful_initialization_loads_model_once(self):
        engine, voice, loader, _ = self.create_engine()

        self.assertTrue(engine.is_loaded)
        self.assertIs(engine._voice, voice)
        self.assertEqual(engine.model_load_count, 1)
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(engine.load())
        self.assertEqual(loader.call_count, 1)

    def test_empty_text_returns_empty_audio_without_synthesis(self):
        engine, voice, _, player = self.create_engine()

        audio = engine.synthesize("  \n")

        self.assertEqual(audio.samples.size, 0)
        self.assertEqual(audio.duration_seconds, 0.0)
        self.assertEqual(voice.synthesis_calls, [])
        self.assertEqual(engine.last_synthesis_metrics["real_time_factor"], 0.0)
        player.play.assert_not_called()

    def test_synthesis_errors_are_reported(self):
        voice = Mock()
        voice.synthesize_wav.side_effect = ValueError("backend failure")
        engine, _, _, _ = self.create_engine(voice_loader=Mock(return_value=voice))

        with self.assertLogs("pi_assistant.voice.tts.piper", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "Piper speech synthesis failed"):
                engine.synthesize("Hello")

    def test_repeated_synthesis_reuses_loaded_model_and_reports_metrics(self):
        engine, voice, loader, _ = self.create_engine()

        first = engine.synthesize("Hello! I'm your personal assistant.")
        second = engine.synthesize("Hello again.")

        self.assertEqual(first.sample_rate, 22050)
        self.assertEqual(first.samples.dtype, np.int16)
        self.assertEqual(first.samples.size, 4)
        self.assertAlmostEqual(first.duration_seconds, 4 / 22050)
        self.assertEqual(second.samples.size, 4)
        self.assertEqual(voice.synthesis_calls, ["Hello! I'm your personal assistant.", "Hello again."])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(engine.model_load_count, 1)
        metrics = engine.last_synthesis_metrics
        self.assertEqual(metrics["model_name"], "voice")
        self.assertEqual(metrics["voice"], "voice")
        self.assertGreaterEqual(metrics["synthesis_time_seconds"], 0)
        self.assertGreater(metrics["audio_duration_seconds"], 0)
        self.assertGreaterEqual(metrics["real_time_factor"], 0)

    def test_speak_uses_audio_player_and_stop_stops_playback(self):
        engine, _, _, player = self.create_engine()

        audio = engine.speak("Hello")
        engine.stop()

        player.play.assert_called_once_with(audio)
        player.stop.assert_called_once_with()

    def test_invalid_piper_audio_is_reported(self):
        voice = FakePiperVoice()

        def write_unsupported_wav(text, wav_file):
            self.assertEqual(text, "Hello")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00")

        voice.synthesize_wav = write_unsupported_wav
        engine, _, _, _ = self.create_engine(voice_loader=Mock(return_value=voice))

        with self.assertLogs("pi_assistant.voice.tts.piper", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "unsupported audio format"):
                engine.synthesize("Hello")


if __name__ == "__main__":
    unittest.main()
