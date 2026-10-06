"""Developer panel UI overlay for previewing and testing assistant states and telemetry."""

from typing import Any, Callable, Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.states import AssistantState
from .developer_controller import DeveloperController
from .ai_test_console import AITestConsole


class DeveloperPanel(QFrame):
    """Semi-transparent developer overlay panel providing manual state triggers and telemetry."""

    def __init__(
        self,
        controller: DeveloperController,
        generate_response: Callable[[str], Tuple[str, Dict[str, Any]]],
        parent: QWidget = None,
        speak_response: Optional[Callable[[str], bool]] = None,
        get_tts_metrics: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        super().__init__(parent)
        self._controller = controller
        self._current_state = AssistantState.IDLE
        self._state_buttons = {}

        self._init_ui()
        self.ai_test_console = AITestConsole(
            generate_response,
            self,
            speak_response=speak_response,
            get_tts_metrics=get_tts_metrics,
        )
        self.layout().insertWidget(6, self.ai_test_console)
        self._connect_signals()

    def _init_ui(self):
        """Construct the developer panel controls and layout."""
        self.setObjectName("DeveloperPanel")
        self.setStyleSheet("""
            QFrame#DeveloperPanel {
                background-color: rgba(15, 23, 42, 0.94);
                border: 1px solid rgba(56, 189, 248, 0.4);
                border-radius: 8px;
            }
            QLabel {
                color: #CBD5E1;
                font-size: 10px;
            }
            QLabel#DevTitle {
                color: #38BDF8;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1px;
            }
            QLabel#DevCurrentState {
                color: #F8FAFC;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#LastTranscriptLabel {
                color: #38BDF8;
                font-size: 10px;
                font-weight: 500;
                background-color: rgba(30, 41, 59, 0.7);
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton#CloseBtn {
                background-color: transparent;
                color: #94A3B8;
                border: none;
                font-size: 12px;
                font-weight: bold;
                padding: 0 4px;
                max-width: 18px;
                max-height: 18px;
            }
            QPushButton#CloseBtn:hover {
                color: #F43F5E;
            }
            QPushButton.StateBtn {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 10px;
                font-weight: 600;
                min-height: 22px;
            }
            QPushButton.StateBtn:hover {
                background-color: #334155;
                border-color: #64748B;
            }
            QPushButton#DevTalkBtn {
                background-color: #059669;
                color: #FFFFFF;
                border: 1px solid #10B981;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
                min-height: 22px;
            }
            QPushButton#DevTalkBtn:hover {
                background-color: #10B981;
            }
            QLabel#TelemetryText {
                color: #94A3B8;
                font-size: 9px;
                font-family: monospace;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header bar: Title, Current State, Close button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        title_label = QLabel("🛠 DEV MODE", self)
        title_label.setObjectName("DevTitle")
        header_layout.addWidget(title_label)

        header_layout.addStretch(1)

        self.state_info_label = QLabel("Current: IDLE", self)
        self.state_info_label.setObjectName("DevCurrentState")
        header_layout.addWidget(self.state_info_label)

        header_layout.addSpacing(6)

        close_btn = QPushButton("✕", self)
        close_btn.setObjectName("CloseBtn")
        close_btn.setToolTip("Close Developer Mode (F12 / ESC)")
        close_btn.clicked.connect(lambda: self._controller.set_mode(False))
        header_layout.addWidget(close_btn)

        layout.addLayout(header_layout)

        # STT, VAD & Hardware Telemetry Row
        hud_row = QHBoxLayout()
        hud_row.setContentsMargins(0, 0, 0, 0)
        hud_row.setSpacing(6)

        self.stt_info_label = QLabel("STT: faster-whisper", self)
        self.stt_info_label.setObjectName("TelemetryText")
        hud_row.addWidget(self.stt_info_label)

        self.vad_info_label = QLabel("VAD: 0.5s", self)
        self.vad_info_label.setObjectName("TelemetryText")
        hud_row.addWidget(self.vad_info_label)

        hud_row.addStretch(1)

        self.mic_status_label = QLabel("Mic: READY", self)
        self.mic_status_label.setObjectName("TelemetryText")
        hud_row.addWidget(self.mic_status_label)

        layout.addLayout(hud_row)

        # Performance Timing Telemetry Row
        timing_row = QHBoxLayout()
        timing_row.setContentsMargins(0, 0, 0, 0)
        timing_row.setSpacing(8)

        self.timing_label = QLabel("Rec: 0.0s | Tail: 0.0s | STT: 0.0s | Total: 0.0s", self)
        self.timing_label.setObjectName("TelemetryText")
        self.timing_label.setWordWrap(True)
        timing_row.addWidget(self.timing_label)

        hud_row.addStretch(1)

        layout.addLayout(timing_row)

        # LLM status and response section
        self.llm_status_label = QLabel("LLM: development fallback", self)
        self.llm_status_label.setObjectName("TelemetryText")
        layout.addWidget(self.llm_status_label)

        self.llm_metrics_label = QLabel("LLM: model=-- | threads=-- | ctx=-- | out=--", self)
        self.llm_metrics_label.setObjectName("TelemetryText")
        layout.addWidget(self.llm_metrics_label)

        self.latency_budget_label = QLabel(
            "Latency | STT: unavailable | LLM: unavailable | TTS: unavailable\n"
            "Post-speech-to-audio-handoff total: unavailable | Target: <5.00s",
            self,
        )
        self.latency_budget_label.setObjectName("TelemetryText")
        self.latency_budget_label.setWordWrap(True)
        layout.addWidget(self.latency_budget_label)

        # Last Recognized Transcription Display
        self.transcript_label = QLabel("Last STT: (None yet - Press TALK)", self)
        self.transcript_label.setObjectName("LastTranscriptLabel")
        self.transcript_label.setWordWrap(True)
        layout.addWidget(self.transcript_label)

        self.response_label = QLabel("Last LLM: (No response yet)", self)
        self.response_label.setObjectName("LastTranscriptLabel")
        self.response_label.setWordWrap(True)
        layout.addWidget(self.response_label)

        # State Selection Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(4)

        states_config = [
            (AssistantState.IDLE, "IDLE (1)"),
            (AssistantState.LISTENING, "LISTEN (2)"),
            (AssistantState.THINKING, "THINK (3)"),
            (AssistantState.SPEAKING, "SPEAK (4)"),
            (AssistantState.ERROR, "ERROR (5)"),
        ]

        for state, label in states_config:
            btn = QPushButton(label, self)
            btn.setProperty("class", "StateBtn")
            btn.clicked.connect(lambda checked=False, s=state: self._controller.request_state(s))
            buttons_layout.addWidget(btn)
            self._state_buttons[state] = btn

        # Quick Talk trigger in Dev Panel
        self.dev_talk_btn = QPushButton("🎙 TALK", self)
        self.dev_talk_btn.setObjectName("DevTalkBtn")
        self.dev_talk_btn.clicked.connect(self._controller.request_interaction)
        buttons_layout.addWidget(self.dev_talk_btn)

        layout.addLayout(buttons_layout)

        # Telemetry & Status bar
        telemetry_layout = QHBoxLayout()
        telemetry_layout.setContentsMargins(0, 0, 0, 0)
        telemetry_layout.setSpacing(8)

        self.fps_label = QLabel("FPS: 60", self)
        self.fps_label.setObjectName("TelemetryText")
        telemetry_layout.addWidget(self.fps_label)

        self.anim_label = QLabel("Anim: Idle", self)
        self.anim_label.setObjectName("TelemetryText")
        telemetry_layout.addWidget(self.anim_label)

        telemetry_layout.addStretch(1)

        shortcuts_tip = QLabel("1-5: States | T: Talk | Space: Toggle", self)
        shortcuts_tip.setObjectName("TelemetryText")
        telemetry_layout.addWidget(shortcuts_tip)

        layout.addLayout(telemetry_layout)

        self._update_button_highlight(AssistantState.IDLE)

    def _connect_signals(self):
        """Connect signals from the developer controller."""
        self._controller.fps_updated.connect(self.update_fps)

    def set_stt_engine_name(self, engine_name: str):
        """Update displayed STT engine name."""
        self.stt_info_label.setText(f"STT: {engine_name}")

    def set_vad_config(self, silence_duration: float):
        """Update displayed VAD silence duration."""
        self.vad_info_label.setText(f"VAD: {silence_duration:.2f}s")

    def set_mic_status(self, status: str, duration: float = 0.0, speech_detected: bool = False):
        """Update microphone status, duration, and speech detection state."""
        speech_state = "Yes" if speech_detected else "No"
        self.mic_status_label.setText(f"Mic: {status} | Speech: {speech_state} | Rec: {duration:.1f}s")

    def set_timing_telemetry(
        self,
        rec_duration: float,
        stt_time: float,
        total_time: float,
        silence_tail: float = 0.0,
        time_to_speech: float = 0.0,
        capture_time: float = 0.0,
    ):
        """Update timing and latency performance measurements."""
        speech_start = f"{time_to_speech:.2f}s" if time_to_speech > 0.0 else "--"
        self.timing_label.setText(
            f"Rec: {rec_duration:.2f}s | Start: {speech_start} | Tail: {silence_tail:.2f}s\n"
            f"Capture: {capture_time:.2f}s | STT: {stt_time:.2f}s | Total: {total_time:.2f}s"
        )

    def set_last_transcription(self, text: str):
        """Update recognized transcription text display."""
        if text.strip():
            self.transcript_label.setText(f'Last STT: "{text.strip()}"')
        else:
            self.transcript_label.setText("Last STT: (No speech detected)")

    def set_llm_status(self, model_name: str, model_path: str = "", threads: str = "", context_size: str = "", max_tokens: str = "", load_time: float = 0.0):
        """Update the developer HUD with LLM configuration and timing."""
        safe_path = model_path.strip()
        if safe_path and len(safe_path) > 24:
            safe_path = "..." + safe_path[-20:]
        self.llm_status_label.setText(
            f"LLM: {model_name} | model={safe_path or 'n/a'} | threads={threads or 'n/a'} | ctx={context_size or 'n/a'} | out={max_tokens or 'n/a'}"
        )
        self.llm_metrics_label.setText(f"LLM load: {load_time:.3f}s | prompt/gen metrics update on response")

    def set_latency_budget(self, metrics: Dict[str, Optional[float]]):
        """Display measured response stages without filling unavailable timings."""
        def format_time(value: Optional[float]) -> str:
            return f"{value:.2f}s" if value is not None else "unavailable"

        self.latency_budget_label.setText(
            f"Latency | STT: {format_time(metrics.get('stt_seconds'))} | "
            f"LLM: {format_time(metrics.get('llm_seconds'))} | "
            f"TTS: {format_time(metrics.get('tts_seconds'))}\n"
            f"Post-speech-to-audio-handoff total: {format_time(metrics.get('total_seconds'))} | "
            f"Target: <{format_time(metrics.get('target_seconds'))}"
        )

    def set_last_response(self, text: str):
        """Display the latest LLM-generated response."""
        if text.strip():
            self.response_label.setText(f'Last LLM: "{text.strip()}"')
        else:
            self.response_label.setText("Last LLM: (No response yet)")

    def set_error_message(self, error_msg: str):
        """Display error alert in Developer Mode HUD."""
        self.transcript_label.setText(f"⚠️ Error: {error_msg}")

    def set_current_state(self, state: AssistantState):
        """Update current displayed state and highlight corresponding button."""
        self._current_state = state
        self.state_info_label.setText(f"Current: {state.name}")
        self._update_button_highlight(state)

        anim_desc = {
            AssistantState.IDLE: "Idle (Blink Active)",
            AssistantState.LISTENING: "Listening (Pulse Active)",
            AssistantState.THINKING: "Thinking (Gaze Shift)",
            AssistantState.SPEAKING: "Speaking (Visemes)",
            AssistantState.ERROR: "Error (Recovering)",
        }
        self.set_animation_status(anim_desc.get(state, state.name))

    def _update_button_highlight(self, active_state: AssistantState):
        """Highlight active button visually."""
        for state, btn in self._state_buttons.items():
            if state == active_state:
                if state == AssistantState.ERROR:
                    btn.setStyleSheet("background-color: #DC2626; color: #FFFFFF; border: 1px solid #F87171;")
                elif state == AssistantState.LISTENING:
                    btn.setStyleSheet("background-color: #0284C7; color: #FFFFFF; border: 1px solid #38BDF8;")
                elif state == AssistantState.THINKING:
                    btn.setStyleSheet("background-color: #D97706; color: #FFFFFF; border: 1px solid #FBBF24;")
                elif state == AssistantState.SPEAKING:
                    btn.setStyleSheet("background-color: #2563EB; color: #FFFFFF; border: 1px solid #60A5FA;")
                else:
                    btn.setStyleSheet("background-color: #0284C7; color: #FFFFFF; border: 1px solid #38BDF8;")
            else:
                btn.setStyleSheet("")

    def update_fps(self, fps: float):
        """Update FPS label with calculated frames per second."""
        self.fps_label.setText(f"FPS: {fps:.0f}")

    def set_animation_status(self, status: str):
        """Update animation status text."""
        self.anim_label.setText(f"Anim: {status}")
