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
        self.tts = tts_engine or TextToSpeechEngine(parent=self)
        self._response_lock = threading.Lock()

        self._last_transcription = ""
        self._last_response = ""
        self._last_llm_error = ""
        self._interaction_start_time = 0.0
        self._recording_duration = 0.0
        self._stt_processing_time = 0.0
        self._total_interaction_time = 0.0
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
        if model_path:
            try:
                return LocalResponseEngine()
            except Exception as exc:
                logger.warning("[AI] Falling back to development response engine because local model is unavailable: %s", exc)
        return DevelopmentResponseEngine()

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

        self.mic.audio_captured.connect(self._on_audio_captured)
        self.mic.error_occurred.connect(self._on_mic_error)

        self.tts.speech_started.connect(self._on_tts_started)
        self.tts.speech_finished.connect(self._on_tts_finished)
        self.tts.error_occurred.connect(self._on_tts_error)

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

    def start_speaking(self, text: str):
        """Transition to SPEAKING and synthesize speech output."""
        logger.info("[Assistant] State -> SPEAKING")
        self.state_manager.set_state(AssistantState.SPEAKING)
        logger.info("[TTS] Speaking: '%s'", text)
        self.tts.speak(text)

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
                response = self.ai.generate_response(text)
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

    def _on_worker_response_done(self, response: str):
        """Handle generated LLM response on the main Qt thread."""
        self.response_ready.emit(response)
        if response and response.strip():
            self._last_response = response
        self._idle_return_timer.start()

    def _on_tts_started(self):
        """Handle TTS playback started."""
        self.state_manager.set_state(AssistantState.SPEAKING)

    def _on_tts_finished(self):
        """Handle TTS playback finished."""
        logger.info("[Assistant] State -> IDLE")
        self.state_manager.set_state(AssistantState.IDLE)

    def _on_mic_error(self, err_msg: str):
        """Handle microphone errors."""
        self._handle_error(err_msg)

    def _on_tts_error(self, err_msg: str):
        """Handle TTS errors."""
        self._handle_error(err_msg)

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
