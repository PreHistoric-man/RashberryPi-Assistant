"""Offline Piper text-to-speech implementation."""

import io
import logging
import os
from pathlib import Path
import threading
import time
import wave
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from PySide6.QtCore import QObject, Signal

from voice.text_to_speech import SpeechAudio
from .audio_player import SoundDeviceAudioPlayer

logger = logging.getLogger("pi_assistant.voice.tts.piper")

VoiceLoader = Callable[[str], Any]


class PiperTextToSpeechEngine(QObject):
    """CPU-based Piper synthesizer that loads and reuses one voice model."""

    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        voice: Optional[str] = None,
        device: Optional[str] = None,
        audio_device: Optional[Union[int, str]] = None,
        *,
        voice_loader: Optional[VoiceLoader] = None,
        audio_player: Optional[Any] = None,
        parent=None,
    ):
        super().__init__(parent)
        configured_path = model_path or os.getenv("PI_ASSISTANT_TTS_MODEL_PATH")
        if configured_path is None or not str(configured_path).strip():
            raise ValueError(
                "Piper model path is not configured. Set "
                "PI_ASSISTANT_TTS_MODEL_PATH to the Piper .onnx model file."
            )

        self.model_path = Path(configured_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Piper model file not found at '{self.model_path}'. "
                "Set PI_ASSISTANT_TTS_MODEL_PATH to the correct .onnx file."
            )

        self.voice_name = (
            (voice or os.getenv("PI_ASSISTANT_TTS_VOICE") or self.model_path.stem).strip()
            or self.model_path.stem
        )
        self.device = self._resolve_inference_device(device)
        self.audio_device = self._resolve_audio_device(audio_device)
        self._voice_loader = voice_loader
        self._audio_player = (
            audio_player if audio_player is not None else SoundDeviceAudioPlayer(device=self.audio_device)
        )
        self._voice: Optional[Any] = None
        self._load_lock = threading.RLock()
        self._load_time_seconds = 0.0
        self._model_load_count = 0
        self._last_synthesis_metrics: Dict[str, Any] = {
            "model_name": self.model_path.stem,
            "voice": self.voice_name,
            "load_time_seconds": 0.0,
            "synthesis_time_seconds": 0.0,
            "audio_duration_seconds": 0.0,
            "real_time_factor": 0.0,
        }

        self.load()

    @staticmethod
    def _resolve_inference_device(device: Optional[str]) -> str:
        """Resolve the Piper inference device, defaulting to CPU."""
        configured_device = (device or os.getenv("PI_ASSISTANT_TTS_DEVICE") or "cpu").strip().lower()
        if configured_device not in {"cpu", "cuda"}:
            raise ValueError("PI_ASSISTANT_TTS_DEVICE must be either 'cpu' or 'cuda'.")
        return configured_device

    @staticmethod
    def _resolve_audio_device(device: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        """Resolve the optional sounddevice output device."""
        configured_device = device
        if configured_device is None:
            configured_device = os.getenv("PI_ASSISTANT_AUDIO_OUTPUT_DEVICE")
        if configured_device is None or not str(configured_device).strip():
            return None
        if isinstance(configured_device, str) and configured_device.strip().isdigit():
            return int(configured_device.strip())
        return configured_device

    @property
    def model_name(self) -> str:
        """Return the configured Piper model identifier."""
        return self.model_path.stem

    @property
    def is_loaded(self) -> bool:
        """Return whether the voice model is loaded."""
        return self._voice is not None

    @property
    def model_load_count(self) -> int:
        """Return the number of successful model loads for this instance."""
        return self._model_load_count

    @property
    def last_synthesis_metrics(self) -> Dict[str, Any]:
        """Return metrics from the most recent synthesis attempt."""
        return dict(self._last_synthesis_metrics)

    def load(self) -> bool:
        """Load the Piper voice once; subsequent calls reuse the same model."""
        with self._load_lock:
            if self._voice is not None:
                return True

            loader = self._voice_loader
            if loader is None:
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise RuntimeError(
                        "piper-tts is required for offline speech synthesis; "
                        "install it with pip install piper-tts."
                    ) from exc
                loader = lambda path: PiperVoice.load(path, use_cuda=self.device == "cuda")

            start = time.perf_counter()
            try:
                logger.info("[TTS] Loading Piper voice model from '%s'", self.model_path)
                self._voice = loader(str(self.model_path))
                self._load_time_seconds = time.perf_counter() - start
                self._model_load_count += 1
                self._last_synthesis_metrics["load_time_seconds"] = self._load_time_seconds
                logger.info("[TTS] Piper voice '%s' loaded in %.3fs", self.voice_name, self._load_time_seconds)
                return True
            except Exception as exc:
                self._voice = None
                raise RuntimeError(f"Piper voice model failed to load: {exc}") from exc

    def synthesize(self, text: str) -> SpeechAudio:
        """Convert text to mono PCM audio without starting playback."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned_text = text.strip()
        if not cleaned_text:
            return self._empty_audio()

        self.load()
        start = time.perf_counter()
        try:
            with self._load_lock:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav_file:
                    voice_config = getattr(self._voice, "config", None)
                    sample_rate = int(getattr(voice_config, "sample_rate", 22050))
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    self._voice.synthesize_wav(cleaned_text, wav_file)
                audio = self._decode_wav(buffer.getvalue())
        except Exception as exc:
            logger.exception("[TTS] Piper synthesis failed")
            self._last_synthesis_metrics = {
                "model_name": self.model_name,
                "voice": self.voice_name,
                "load_time_seconds": self._load_time_seconds,
                "synthesis_time_seconds": time.perf_counter() - start,
                "audio_duration_seconds": 0.0,
                "real_time_factor": 0.0,
                "error": str(exc),
            }
            raise RuntimeError(f"Piper speech synthesis failed: {exc}") from exc

        synthesis_time = time.perf_counter() - start
        audio_duration = audio.duration_seconds
        self._last_synthesis_metrics = {
            "model_name": self.model_name,
            "voice": self.voice_name,
            "load_time_seconds": self._load_time_seconds,
            "synthesis_time_seconds": synthesis_time,
            "audio_duration_seconds": audio_duration,
            "real_time_factor": synthesis_time / audio_duration if audio_duration > 0 else 0.0,
        }
        logger.info(
            "[TTS] Synthesized %.3fs of audio in %.3fs (RTF %.3f)",
            audio_duration,
            synthesis_time,
            self._last_synthesis_metrics["real_time_factor"],
        )
        return audio

    def speak(self, text: str) -> SpeechAudio:
        """Synthesize text and play it using the configured audio output."""
        speech_start = time.perf_counter()
        self.speech_started.emit()
        try:
            audio = self.synthesize(text)
            self._last_synthesis_metrics["playback_handoff_seconds"] = (
                time.perf_counter() - speech_start
            )
            self._audio_player.play(audio)
            return audio
        except Exception as exc:
            self.error_occurred.emit(f"TTS synthesis error: {exc}")
            raise
        finally:
            self.speech_finished.emit()

    def stop(self) -> None:
        """Stop audio playback."""
        self._audio_player.stop()

    def _empty_audio(self) -> SpeechAudio:
        voice_config = getattr(self._voice, "config", None)
        sample_rate = getattr(voice_config, "sample_rate", 22050)
        self._last_synthesis_metrics = {
            "model_name": self.model_name,
            "voice": self.voice_name,
            "load_time_seconds": self._load_time_seconds,
            "synthesis_time_seconds": 0.0,
            "audio_duration_seconds": 0.0,
            "real_time_factor": 0.0,
        }
        return SpeechAudio(samples=np.empty(0, dtype=np.int16), sample_rate=int(sample_rate))

    @staticmethod
    def _decode_wav(wav_data: bytes) -> SpeechAudio:
        """Validate Piper's WAV output and convert its PCM frames to NumPy."""
        try:
            with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                frames = wav_file.readframes(frame_count)
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"Piper returned invalid WAV audio: {exc}") from exc

        if channels < 1 or sample_rate < 1 or sample_width != 2:
            raise ValueError(
                "Piper returned unsupported audio format "
                f"(channels={channels}, sample_width={sample_width}, sample_rate={sample_rate})."
            )
        samples = np.frombuffer(frames, dtype="<i2").copy()
        if samples.size != frame_count * channels:
            raise ValueError("Piper returned truncated WAV audio.")
        if channels > 1:
            samples = samples.reshape(frame_count, channels)
        return SpeechAudio(samples=samples, sample_rate=sample_rate)


__all__ = ["PiperTextToSpeechEngine"]
