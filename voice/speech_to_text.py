"""Speech-to-Text abstraction and exports."""

from typing import Protocol, Union
import io
import numpy as np

from .stt.faster_whisper_stt import FasterWhisperSTT


class BaseSpeechToTextEngine(Protocol):
    """Protocol defining the Speech-to-Text interface."""

    def transcribe(self, audio_data: Union[bytes, io.BytesIO, np.ndarray]) -> str:
        """Transcribe audio WAV bytes or numpy audio into text."""
        ...


__all__ = ["BaseSpeechToTextEngine", "FasterWhisperSTT"]
