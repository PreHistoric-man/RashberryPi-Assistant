"""Assistant Core orchestrating the interaction lifecycle, audio pipeline, and states."""

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal

from ai.response_engine import BaseResponseEngine, DevelopmentResponseEngine, LocalResponseEngine
from core.state_manager import StateManager
from ui.states import AssistantState
from voice.microphone import MicrophoneRecorder
from voice.speech_to_text import BaseSpeechToTextEngine, FasterWhisperSTT
from voice.text_to_speech import TextToSpeechEngine
from voice.tts.piper_tts import PiperTextToSpeechEngine

logger = logging.getLogger("pi_assistant.core")


class AssistantCore(QObject):
    """Central orchestrator managing voice interaction lifecycle and component dispatch."""

    interaction_started = Signal()
    transcription_ready = Signal(str)
    telemetry_updated = Signal(float, float, float)  # (rec_duration, stt_time, total_time)
    response_ready = Signal(str)
    error_occurred = Signal(str)

    # Internal signals for safe cross-thread Qt slot dispatch
    _worker_transcription_done = Signal(str, float, float, float)
    _worker_response_done = Signal(str)
    _worker_error_occurred = Signal(str)
    _worker_tts_started = Signal()
    _worker_tts_finished = Signal()
    _worker_tts_failed = Signal(str)

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        mic_recorder: Optional[MicrophoneRecorder] = None,
        stt_engine: Optional[BaseSpeechToTextEngine] = None,
        response_engine: Optional[BaseResponseEngine] = None,
        tts_engine: Optional[TextToSpeechEngine] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.state_manager = state_manager or StateManager(parent=self)
        self.mic = mic_recorder or MicrophoneRecorder(parent=self)
        self.stt = stt_engine or FasterWhisperSTT()
        self.ai = response_engine or self._default_response_engine()
        self.tts = tts_engine or self._default_tts_engine(parent=self)
        self._response_lock = threading.Lock()
        self._tts_lock = threading.Lock()

        self._last_transcription = ""
        self._last_response = ""
        self._last_llm_error = ""
        self._interaction_start_time = 0.0
        self._recording_duration = 0.0
        self._stt_processing_time = 0.0
        self._total_interaction_time = 0.0
        self._llm_processing_time: Optional[float] = None
        self._llm_generation_metrics: Dict[str, Any] = {}
        self._last_tts_processing_time: Optional[float] = None
        self._post_capture_start_time = 0.0
        self._tts_worker_start_time = 0.0
        self._post_speech_latency: Optional[float] = None
        self._silence_tail_duration = 0.0
        self._time_until_speech = 0.0

        # Auto-return timer to IDLE after transcription display
        self._idle_return_timer = QTimer(self)
        self._idle_return_timer.setSingleShot(True)
        self._idle_return_timer.setInterval(2800)
        self._idle_return_timer.timeout.connect(self._return_to_idle)

        # Recovery timer for ERROR state
        self._error_recovery_timer = QTimer(self)
        self._error_recovery_timer.setSingleShot(True)
        self._error_recovery_timer.setInterval(2200)
        self._error_recovery_timer.timeout.connect(self._recover_to_idle)

        self._connect_internal_signals()

    @staticmethod
    def _default_response_engine():
        model_path = os.getenv("PI_ASSISTANT_LLM_MODEL_PATH")
        if not model_path:
            return DevelopmentResponseEngine("PI_ASSISTANT_LLM_MODEL_PATH is not set")
        try:
            return LocalResponseEngine()
        except Exception as exc:
            logger.warning("[AI] Falling back to development response engine because local model is unavailable: %s", exc)
            return DevelopmentResponseEngine(f"local model initialization failed: {exc}")

    @staticmethod
    def _default_tts_engine(parent=None):
        model_path = os.getenv("PI_ASSISTANT_TTS_MODEL_PATH")
        if model_path:
            try:
                return PiperTextToSpeechEngine(model_path=model_path, parent=parent)
            except Exception as exc:
                logger.warning("[TTS] Falling back to desktop TTS because Piper is unavailable: %s", exc)
        return TextToSpeechEngine(parent=parent)

    @property
    def last_transcription(self) -> str:
        """Return the most recent recognized transcription text."""
        return self._last_transcription

    @property
    def last_response(self) -> str:
        """Return the most recent LLM response text."""
        return self._last_response

    @property
    def last_llm_error(self) -> str:
        """Return the most recent LLM error string."""
        return self._last_llm_error

    @property
    def stt_engine_name(self) -> str:
        """Return the name of the active STT engine."""
        return getattr(self.stt, "engine_name", "faster-whisper")

    @property
    def vad_silence_duration(self) -> float:
        """Return configured VAD silence duration."""
        return getattr(self.mic, "silence_duration", 0.5)

    @property
    def last_recording_duration(self) -> float:
        """Return recording duration in seconds of the last interaction."""
        return self._recording_duration

    @property
    def last_stt_time(self) -> float:
        """Return STT processing time in seconds of the last interaction."""
        return self._stt_processing_time

    @property
    def last_total_time(self) -> float:
        """Return total interaction time in seconds of the last interaction."""
        return self._total_interaction_time

    @property
    def last_silence_tail(self) -> float:
        """Return silence tail duration in seconds."""
        return self._silence_tail_duration

    @property
    def last_time_until_speech(self) -> float:
        """Return elapsed time in seconds until speech was detected."""
        return self._time_until_speech

    def _connect_internal_signals(self):
        """Connect voice pipeline events to state and interaction handlers."""
        self._worker_transcription_done.connect(self._on_worker_transcription_done)
        self._worker_response_done.connect(self._on_worker_response_done)
        self._worker_error_occurred.connect(self._handle_error)
        self._worker_tts_started.connect(self._on_tts_started)
        self._worker_tts_finished.connect(self._on_tts_finished)
        self._worker_tts_failed.connect(self._on_worker_tts_failed)

        self.mic.audio_captured.connect(self._on_audio_captured)
        self.mic.error_occurred.connect(self._on_mic_error)

        for signal_name, handler in (
            ("speech_started", self._on_tts_started),
            ("speech_finished", self._on_tts_finished),
            ("error_occurred", self._on_tts_error),
        ):
            signal = getattr(self.tts, signal_name, None)
            if signal is not None:
                signal.connect(handler)

    # --- Interaction Lifecycle API ---

    def start_interaction(self):
        """Trigger the standard voice interaction cycle (TALK button / GPIO trigger)."""
        logger.info("[Assistant] Interaction started")
        self._interaction_start_time = time.perf_counter()
        self._idle_return_timer.stop()
        self._error_recovery_timer.stop()
        self.interaction_started.emit()
        self.start_listening()

    def start_listening(self):
        """Transition to LISTENING and initiate microphone audio capture."""
        if self.state_manager.current_state == AssistantState.SPEAKING:
            self.tts.stop()

        logger.info("[Assistant] State -> LISTENING")
        self.state_manager.set_state(AssistantState.LISTENING)
        self.mic.start_recording()

    def process_input(self, text: str):
        """Process transcribed user input through AI."""
        logger.info("[Assistant] State -> THINKING")
        self.state_manager.set_state(AssistantState.THINKING)
        logger.info("[Assistant] Input received: '%s'", text)

    def start_speaking(self, text: str) -> bool:
        """Start speech through the shared TTS engine without blocking the UI."""
        cleaned = (text or "").strip()
        if not cleaned:
            return False

        if not self._tts_lock.acquire(blocking=False):
            logger.info("[TTS] Ignoring speech request because speech is already active.")
            return False

        self._idle_return_timer.stop()

        logger.info("[TTS] Speaking: '%s'", cleaned)

        if hasattr(self.tts, "speech_started") and hasattr(self.tts, "speech_finished"):
            threading.Thread(target=self._tts_worker, args=(cleaned,), daemon=True).start()
            return True

        threading.Thread(target=self._tts_worker_fallback, args=(cleaned,), daemon=True).start()
        return True

    def _tts_worker(self, text: str):
        """Run TTS synthesis/playback in a background worker while keeping the UI responsive."""
        try:
            self._tts_worker_start_time = time.perf_counter()
            self.tts.speak(text)
        except Exception as exc:
            logger.exception("[TTS] Speech synthesis failed")
            if not hasattr(self.tts, "error_occurred"):
                self._worker_tts_failed.emit(f"TTS synthesis error: {exc}")

    def _tts_worker_fallback(self, text: str):
        """Handle TTS engines without built-in lifecycle signals."""
        self._worker_tts_started.emit()
        try:
            self._tts_worker_start_time = time.perf_counter()
            self.tts.speak(text)
        except Exception as exc:
            logger.exception("[TTS] Speech synthesis failed")
            if not hasattr(self.tts, "error_occurred"):
                self._worker_tts_failed.emit(f"TTS synthesis error: {exc}")
        else:
            self._worker_tts_finished.emit()

    def stop(self):
        """Halt all active operations and return to IDLE."""
        logger.info("[Assistant] Stopping all active tasks.")
        self.mic.stop_recording()
        self.tts.stop()
        self._idle_return_timer.stop()
        self._error_recovery_timer.stop()
        logger.info("[Assistant] State -> IDLE")
        self.state_manager.set_state(AssistantState.IDLE)

    # --- Internal Event Handlers & Workers ---

    def _on_audio_captured(self, wav_bytes: bytes):
        """Handle recorded audio buffer from microphone."""
        self._post_capture_start_time = time.perf_counter()
        self._tts_worker_start_time = 0.0
        self._llm_processing_time = None
        self._llm_generation_metrics = {}
        self._last_tts_processing_time = None
        self._post_speech_latency = None
        self._recording_duration = self.mic.last_duration
        self._silence_tail_duration = getattr(self.mic, "silence_tail_duration", 0.0)
        self._time_until_speech = getattr(self.mic, "time_until_speech_detected", 0.0)

        logger.info("[Assistant] State -> THINKING")
        self.state_manager.set_state(AssistantState.THINKING)

        threading.Thread(
            target=self._transcribe_worker,
            args=(wav_bytes,),
            daemon=True,
        ).start()

    def _transcribe_worker(self, wav_bytes: bytes):
        """Worker thread for Whisper transcription."""
        try:
            stt_start_time = time.perf_counter()
            text = self.stt.transcribe(wav_bytes)
            stt_processing_time = time.perf_counter() - stt_start_time
            total_time = (
                time.perf_counter() - self._interaction_start_time
                if self._interaction_start_time > 0
                else stt_processing_time
            )

            self._worker_transcription_done.emit(
                text,
                self._recording_duration,
                stt_processing_time,
                total_time,
            )

        except Exception as e:
            logger.error("[STT] Transcription worker exception: %s", e)
            self._worker_error_occurred.emit(f"Speech recognition error: {e}")

    def _on_worker_transcription_done(
        self,
        text: str,
        rec_duration: float,
        stt_processing_time: float,
        total_time: float,
    ):
        """Handle transcription completion on the main Qt thread."""
        self._last_transcription = text
        self._recording_duration = rec_duration
        self._stt_processing_time = stt_processing_time
        self._total_interaction_time = total_time

        self.transcription_ready.emit(text)
        self.telemetry_updated.emit(rec_duration, stt_processing_time, total_time)

        if text and text.strip():
            threading.Thread(
                target=self._generate_response_worker,
                args=(text,),
                daemon=True,
            ).start()
            return

        self._idle_return_timer.start()

    def _generate_response_worker(self, text: str):
        """Worker thread for AI response generation without blocking the UI loop."""
        try:
            if not hasattr(self.ai, "generate_response"):
                self._worker_response_done.emit("")
                return

            with self._response_lock:
                llm_start_time = time.perf_counter()
                response = self.ai.generate_response(text)
                self._llm_processing_time = time.perf_counter() - llm_start_time
                self._llm_generation_metrics = dict(
                    getattr(self.ai, "last_generation_metrics", {}) or {}
                )
            self._last_response = response
            self._last_llm_error = ""
            logger.info("[AI] Response generated: '%s'", response)
            self._worker_response_done.emit(response)
        except Exception as e:
            self._last_llm_error = str(e)
            logger.error("[AI] Response generation exception: %s", e)
            self._worker_error_occurred.emit(f"AI response generation error: {e}")

    def generate_developer_response(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Generate a developer-console response without entering the voice pipeline."""
        if not hasattr(self.ai, "generate_response"):
            raise RuntimeError("The active response engine does not support text generation.")

        with self._response_lock:
            response = self.ai.generate_response(text)
            metrics = dict(getattr(self.ai, "last_generation_metrics", {}) or {})
        return response, metrics

    def get_tts_metrics(self) -> Dict[str, Any]:
        """Return metrics supplied by the active TTS engine, when available."""
        return dict(getattr(self.tts, "last_synthesis_metrics", {}) or {})

    def get_latency_metrics(self) -> Dict[str, Optional[float]]:
        """Return measured interaction-stage timings, leaving unavailable values unset."""
        llm_time = self._llm_generation_metrics.get("llm_wall_time_seconds")
        return {
            "stt_seconds": self._stt_processing_time if self._post_capture_start_time else None,
            "llm_seconds": float(llm_time) if llm_time is not None else self._llm_processing_time,
            "tts_seconds": self._last_tts_processing_time,
            "total_seconds": self._post_speech_latency,
            "target_seconds": 5.0,
        }

    def _on_worker_response_done(self, response: str):
        """Handle generated LLM response on the main Qt thread."""
        self.response_ready.emit(response)
        if response and response.strip():
            self._last_response = response
            self.start_speaking(response)
            return
        self._idle_return_timer.start()

    def _on_tts_started(self):
        """Handle TTS playback started."""
        self.state_manager.set_state(AssistantState.SPEAKING)

    def _on_tts_finished(self):
        """Handle TTS playback finished."""
        self._release_tts_lock()
        tts_metrics = self.get_tts_metrics()
        self._post_speech_latency = None
        synthesis_time = tts_metrics.get("synthesis_time_seconds")
        self._last_tts_processing_time = (
            float(synthesis_time) if synthesis_time is not None else None
        )
        handoff_time = tts_metrics.get("playback_handoff_seconds")
        if (
            handoff_time is not None
            and self._tts_worker_start_time > 0.0
            and self._post_capture_start_time > 0.0
        ):
            self._post_speech_latency = (
                self._tts_worker_start_time
                + float(handoff_time)
                - self._post_capture_start_time
                + self._silence_tail_duration
            )
        if self.state_manager.current_state == AssistantState.ERROR:
            return
        logger.info("[Assistant] State -> IDLE")
        self.state_manager.set_state(AssistantState.IDLE)

    def _on_mic_error(self, err_msg: str):
        """Handle microphone errors."""
        self._handle_error(err_msg)

    def _on_tts_error(self, err_msg: str):
        """Handle TTS errors."""
        self._release_tts_lock()
        self._handle_error(err_msg)

    def _on_worker_tts_failed(self, err_msg: str):
        """Handle exceptions from TTS engines that do not emit an error signal."""
        self._release_tts_lock()
        self._handle_error(err_msg)

    def _release_tts_lock(self):
        """Release the speech reservation after the TTS lifecycle completes."""
        if self._tts_lock.locked():
            self._tts_lock.release()

    def _return_to_idle(self):
        """Safely return the assistant to IDLE after processing is complete."""
        if self.state_manager.current_state != AssistantState.ERROR:
            logger.info("[Assistant] State -> IDLE")
            self.state_manager.set_state(AssistantState.IDLE)

    def _recover_to_idle(self):
        """Recover from ERROR state back to IDLE."""
        logger.info("[Assistant] State -> IDLE")
        self.state_manager.set_state(AssistantState.IDLE)

    def _handle_error(self, err_msg: str):
        """Transition to ERROR state, emit notification, and schedule auto-recovery."""
        logger.error("[Assistant] Error: %s", err_msg)
        self.error_occurred.emit(err_msg)
        logger.info("[Assistant] State -> ERROR")
        self.state_manager.set_state(AssistantState.ERROR)
        self._error_recovery_timer.start()
