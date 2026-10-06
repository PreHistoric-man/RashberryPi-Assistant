"""Text-to-Speech engine module for AI Assistant voice output."""

from dataclasses import dataclass
import logging
import threading
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("pi_assistant.voice.tts")


@dataclass(frozen=True)
class SpeechAudio:
    """PCM audio returned by a text-to-speech engine."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Return the duration of the audio in seconds."""
        if self.samples.size == 0:
            return 0.0
        return self.samples.shape[0] / self.sample_rate


@runtime_checkable
class BaseTextToSpeechEngine(Protocol):
    """Model-agnostic contract for synthesizing and playing speech."""

    def synthesize(self, text: str) -> SpeechAudio:
        """Convert text to in-memory audio without playing it."""
        ...

    def speak(self, text: str) -> SpeechAudio:
        """Synthesize text and send the resulting audio to playback."""
        ...

    def stop(self) -> None:
        """Stop active playback."""
        ...


class TextToSpeechEngine(QObject):
    """Non-blocking Text-to-Speech engine for local Windows audio output."""

    speech_started = Signal()
    speech_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, voice_rate: int = 175, parent=None):
        super().__init__(parent)
        self.voice_rate = voice_rate
        self._is_speaking = False
        self._speech_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_speaking(self) -> bool:
        """Return True if TTS is currently outputting speech."""
        return self._is_speaking

    def speak(self, text: str):
        """Asynchronously synthesize and play text speech."""
        cleaned = text.strip()
        if not cleaned:
            self.speech_finished.emit()
            return

        with self._lock:
            if self._is_speaking:
                logger.warning("[TTS] Already speaking. Queuing/replacing not supported in simple mode.")
                return
            self._is_speaking = True

        self._speech_thread = threading.Thread(
            target=self._speak_worker,
            args=(cleaned,),
            daemon=True,
        )
        self._speech_thread.start()

    def _speak_worker(self, text: str):
        """Worker executing TTS synthesis in a dedicated background thread."""
        logger.info("[TTS] Speaking: '%s'", text)
        self.speech_started.emit()

        success = False
        try:
            # First attempt using pyttsx3
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.voice_rate)
            engine.say(text)
            engine.runAndWait()
            # Clean up engine to allow reinitialization on next call
            del engine
            success = True
        except Exception as e:
            logger.warning("[TTS] pyttsx3 failed (%s), attempting Windows SAPI fallback...", e)
            try:
                # Windows COM SAPI fallback
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
                del speaker
                success = True
            except Exception as e2:
                logger.error("[TTS] SAPI fallback also failed: %s", e2)
                self.error_occurred.emit(f"TTS synthesis error: {e2}")

        with self._lock:
            self._is_speaking = False

        logger.info("[TTS] Speech playback complete.")
        self.speech_finished.emit()

    def stop(self):
        """Stop any active speech."""
        # For simple non-blocking TTS, thread terminates naturally on completion
        with self._lock:
            self._is_speaking = False


__all__ = [
    "BaseTextToSpeechEngine",
    "SpeechAudio",
    "TextToSpeechEngine",
]
