"""Faster-Whisper Speech-to-Text concrete implementation."""

import io
import logging
import os
from typing import Optional, Union
import numpy as np
import soundfile as sf

logger = logging.getLogger("pi_assistant.voice.stt.faster_whisper")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("[STT] faster-whisper is not installed or failed to import.")


class FasterWhisperSTT:
    """Concrete Speech-to-Text engine powered by faster-whisper.

    Loads the model once and reuses it for high-performance offline transcription.
    Decodes audio to 16kHz float32 NumPy arrays for maximum compatibility and speed.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        download_root: Optional[str] = None,
        cpu_threads: Optional[int] = None,
    ):
        configured_model = os.getenv("PI_ASSISTANT_STT_MODEL", "small").strip()
        self.model_size = model_size or configured_model or "small"
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.download_root = download_root
        self.cpu_threads = self._resolve_cpu_threads(cpu_threads)

        self._model: Optional["WhisperModel"] = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

        # Preload model on startup
        self._load_model()

    @staticmethod
    def _resolve_cpu_threads(cpu_threads: Optional[int]) -> int:
        """Resolve the configured CTranslate2 CPU thread count."""
        if cpu_threads is not None:
            if cpu_threads < 0:
                raise ValueError("cpu_threads must be zero or greater")
            return cpu_threads

        configured_threads = os.getenv("PI_ASSISTANT_STT_CPU_THREADS", "3").strip()
        try:
            resolved_threads = int(configured_threads)
        except ValueError:
            logger.warning("[STT] Invalid PI_ASSISTANT_STT_CPU_THREADS=%r; using 3", configured_threads)
            return 3

        if resolved_threads < 0:
            logger.warning("[STT] CPU thread count must be non-negative; using 3")
            return 3
        return resolved_threads

    @property
    def engine_name(self) -> str:
        """Return human-readable engine identifier."""
        return f"faster-whisper ({self.model_size}, {self.compute_type}, {self.cpu_threads} threads)"

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for transcription."""
        return self._is_loaded and self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        """Return error description if model loading failed."""
        return self._load_error

    def _load_model(self):
        """Load the faster-whisper model into memory (reused across all requests)."""
        if not FASTER_WHISPER_AVAILABLE:
            self._load_error = "faster-whisper library is not installed."
            logger.error("[STT] %s", self._load_error)
            return

        try:
            logger.info(
                "[STT] Loading faster-whisper model '%s' (device=%s, compute=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=self.download_root,
            )
            self._is_loaded = True
            self._load_error = None
            logger.info("[STT] faster-whisper model loaded successfully.")
        except Exception as e:
            # Fallback attempt with default compute type if int8 encounters CPU limitations
            logger.warning("[STT] Failed with compute_type='%s' (%s), trying 'default'...", self.compute_type, e)
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="default",
                    cpu_threads=self.cpu_threads,
                    download_root=self.download_root,
                )
                self._is_loaded = True
                self._load_error = None
                self.compute_type = "default"
                logger.info("[STT] faster-whisper model loaded with fallback compute_type.")
            except Exception as e2:
                self._load_error = f"Failed to load WhisperModel: {e2}"
                logger.error("[STT] Model loading failed: %s", e2)
                self._is_loaded = False
                self._model = None

    def transcribe(self, audio_data: Union[bytes, io.BytesIO, np.ndarray]) -> str:
        """Transcribe audio WAV bytes or numpy audio array into clean text string.

        Args:
            audio_data: Audio data in WAV byte format, BytesIO stream, or float32 1D numpy array.

        Returns:
            str: Clean plain recognized text, or empty string if no speech or error.
        """
        if not self.is_ready:
            if not self._is_loaded and self._load_error is None:
                self._load_model()
            if not self.is_ready:
                logger.error("[STT] Cannot transcribe: model is not loaded (%s)", self._load_error)
                return ""

        if audio_data is None:
            return ""

        try:
            logger.info("[STT] Transcription started")

            # Convert bytes / BytesIO to float32 NumPy array at 16kHz
            if isinstance(audio_data, (bytes, io.BytesIO)):
                stream = io.BytesIO(audio_data) if isinstance(audio_data, bytes) else audio_data
                stream.seek(0)
                audio_np, sr = sf.read(stream, dtype="float32")
                # Ensure mono
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=1)
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data.astype(np.float32)
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=1)
            else:
                logger.error("[STT] Unsupported audio data type: %s", type(audio_data))
                return ""

            if len(audio_np) == 0:
                logger.info("[STT] Empty audio array received.")
                return ""

            # Transcribe numpy float32 audio array directly (fastest & most reliable)
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )

            # Collect transcribed text across segments
            text_segments = []
            for segment in segments:
                cleaned = segment.text.strip()
                if cleaned:
                    text_segments.append(cleaned)

            transcribed_text = " ".join(text_segments).strip()

            logger.info("[STT] Transcription finished")
            logger.info(
                "[STT] Result received: '%s'",
                transcribed_text,
            )
            return transcribed_text

        except Exception as e:
            logger.error("[STT] Transcription error: %s", e)
            return ""
