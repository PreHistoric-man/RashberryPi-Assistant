"""Microphone capture module for AI Assistant voice input."""

import io
import logging
import threading
import time
from typing import Optional, Union
import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("pi_assistant.voice.microphone")

try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except Exception as e:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice or soundfile not available: %s", e)


class MicrophoneRecorder(QObject):
    """Asynchronous microphone recorder using sounddevice with automatic energy-based silence detection."""

    recording_started = Signal()
    speech_detected = Signal()
    silence_detected = Signal()
    status_changed = Signal(str)        # Emits "READY", "RECORDING", "SILENCE DETECTED", "ERROR"
    audio_captured = Signal(bytes)      # Emits WAV audio bytes
    error_occurred = Signal(str)        # Emits error description

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[Union[int, str]] = None,
        silence_threshold_rms: Optional[float] = None,
        silence_duration_seconds: Optional[float] = None,
        max_recording_duration_seconds: Optional[float] = None,
        energy_window_ms: int = 100,
        # Backward compatibility keyword aliases
        silence_threshold: float = 0.015,
        silence_duration: float = 0.5,
        max_duration: float = 8.0,
        parent=None,
    ):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.device = device
        self.energy_window_ms = max(50, min(100, energy_window_ms))

        # Resolution for config aliases (preference to explicit parameter names if provided)
        self.silence_threshold_rms = silence_threshold_rms if silence_threshold_rms is not None else silence_threshold
        self.silence_duration_seconds = (
            silence_duration_seconds if silence_duration_seconds is not None else silence_duration
        )
        self.max_recording_duration_seconds = (
            max_recording_duration_seconds if max_recording_duration_seconds is not None else max_duration
        )
        self.silence_threshold = self.silence_threshold_rms
        self.silence_duration = self.silence_duration_seconds
        self.max_duration = self.max_recording_duration_seconds

        self._is_recording = False
        self._record_thread = None
        self._stop_event = threading.Event()
        self._status = "READY"
        self._capture_started_at = 0.0

        # Telemetry measurements
        self._last_duration = 0.0
        self._time_until_speech = 0.0
        self._silence_tail_duration = 0.0
        self._total_capture_time = 0.0
        self._speech_detected = False

    @property
    def is_recording(self) -> bool:
        """Return True if microphone is currently recording."""
        return self._is_recording

    @property
    def last_duration(self) -> float:
        """Return the duration in seconds of the most recent recording."""
        return self._last_duration

    @property
    def recording_duration(self) -> float:
        """Return current capture time or the duration of the most recent recording."""
        if self._is_recording and self._capture_started_at:
            return time.perf_counter() - self._capture_started_at
        return self._last_duration

    @property
    def time_until_speech_detected(self) -> float:
        """Return time in seconds from recording start until speech was first detected."""
        return self._time_until_speech

    @property
    def silence_tail_duration(self) -> float:
        """Return duration in seconds of trailing silence after speech."""
        return self._silence_tail_duration

    @property
    def total_capture_time(self) -> float:
        """Return total elapsed wall-clock time for the capture session."""
        return self._total_capture_time

    @property
    def speech_detected_in_last_recording(self) -> bool:
        """Return True if speech was detected during the most recent recording."""
        return self._speech_detected

    @property
    def status(self) -> str:
        """Return current microphone status string."""
        return self._status

    def is_available(self) -> bool:
        """Check if an input audio device is available on the host system."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
        try:
            device_info = sd.query_devices(device=self.device, kind="input")
            return device_info is not None and device_info.get("max_input_channels", 0) > 0
        except Exception as e:
            logger.warning("[Microphone] Device query failed: %s", e)
            return False

    def start_recording(self):
        """Start capturing audio in a background thread."""
        if self._is_recording:
            logger.warning("[Microphone] Already recording.")
            return

        if not self.is_available():
            err_msg = "No functional microphone detected or sounddevice unavailable."
            logger.error("[Microphone] %s", err_msg)
            self._status = "ERROR"
            self.status_changed.emit("ERROR")
            self.error_occurred.emit(err_msg)
            return

        self._is_recording = True
        self._stop_event.clear()
        self._capture_started_at = time.perf_counter()
        self._speech_detected = False
        self._time_until_speech = 0.0
        self._silence_tail_duration = 0.0
        self._total_capture_time = 0.0
        self._record_thread = threading.Thread(target=self._record_worker, daemon=True)
        self._record_thread.start()

    def stop_recording(self):
        """Signal recording to stop immediately."""
        if self._is_recording:
            self._stop_event.set()

    def _record_worker(self):
        """Worker thread capturing audio frames with low-latency silence detection."""
        logger.info("[Microphone] Recording started")
        self._status = "RECORDING"
        self.status_changed.emit("RECORDING")
        self.recording_started.emit()

        record_start_time = time.perf_counter()
        frames = []
        chunk_duration = self.energy_window_ms / 1000.0  # Analysis window (e.g. 100ms)
        chunk_samples = max(1, int(self.sample_rate * chunk_duration))
        max_samples = max(1, int(self.sample_rate * self.max_duration))

        silence_threshold = self.silence_threshold_rms
        speech_has_started = False
        silence_elapsed = 0.0
        self._time_until_speech = 0.0
        self._speech_detected = False
        captured_samples = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                device=self.device,
            ) as stream:
                while captured_samples < max_samples:
                    if self._stop_event.is_set():
                        break

                    requested_samples = min(chunk_samples, max_samples - captured_samples)
                    data, overflowed = stream.read(requested_samples)
                    if overflowed:
                        logger.debug("[Microphone] Buffer overflowed.")

                    frames.append(data.copy())
                    samples_read = len(data)
                    captured_samples += samples_read
                    block_duration = samples_read / self.sample_rate

                    # Energy calculation for VAD (Voice Activity Detection)
                    rms = float(np.sqrt(np.mean(data**2)))
                    if rms > silence_threshold:
                        if not speech_has_started:
                            speech_has_started = True
                            self._speech_detected = True
                            self._time_until_speech = time.perf_counter() - record_start_time
                            logger.info(
                                "[Microphone] Speech detected (after %.2fs)",
                                self._time_until_speech,
                            )
                            self.speech_detected.emit()
                        silence_elapsed = 0.0
                    else:
                        if speech_has_started:
                            silence_elapsed += block_duration
                            if silence_elapsed >= self.silence_duration_seconds:
                                logger.info(
                                    "[Microphone] Silence detected (tail: %.2fs)",
                                    silence_elapsed,
                                )
                                self.silence_detected.emit()
                                self._status = "SILENCE DETECTED"
                                self.status_changed.emit("SILENCE DETECTED")
                                break
                        # If speech has not started yet, continue listening through initial silence

        except Exception as e:
            err_msg = f"Microphone error during capture: {e}"
            logger.error("[Microphone] %s", err_msg)
            self._is_recording = False
            self._status = "ERROR"
            self.status_changed.emit("ERROR")
            self.error_occurred.emit(err_msg)
            return

        self._is_recording = False
        self._total_capture_time = time.perf_counter() - record_start_time
        self._silence_tail_duration = silence_elapsed if speech_has_started else 0.0

        logger.info(
            "[Microphone] Recording finished (total: %.2fs, speech_detected=%s, tail=%.2fs)",
            self._total_capture_time,
            speech_has_started,
            self._silence_tail_duration,
        )

        if not frames:
            err_msg = "No audio frames captured."
            logger.warning("[Microphone] %s", err_msg)
            self._status = "READY"
            self.status_changed.emit("READY")
            self.error_occurred.emit(err_msg)
            return

        try:
            # Concatenate and export to WAV bytes
            audio_data = np.concatenate(frames, axis=0)
            self._last_duration = len(audio_data) / self.sample_rate
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, self.sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = wav_buffer.getvalue()

            self._status = "READY"
            self.status_changed.emit("READY")
            logger.info(
                "[Microphone] Audio captured successfully (%d bytes, %.2fs).",
                len(wav_bytes),
                self._last_duration,
            )
            self.audio_captured.emit(wav_bytes)

        except Exception as e:
            err_msg = f"Failed to encode audio WAV data: {e}"
            logger.error("[Microphone] %s", err_msg)
            self._status = "READY"
            self.status_changed.emit("READY")
            self.error_occurred.emit(err_msg)
