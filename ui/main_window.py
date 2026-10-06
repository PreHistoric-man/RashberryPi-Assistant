"""MainWindow for Raspberry Pi AI Assistant UI.

Designed for 3.5-inch TFT LCD displays (480x320 resolution) and scalable for development.
Integrates AssistantCore, StateManager, TALK development button, and Developer Mode overlay.
"""

from typing import Optional, TYPE_CHECKING
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.developer import DeveloperController, DeveloperPanel
from ui.face_widget import FaceWidget
from ui.states import AssistantState

if TYPE_CHECKING:
    from core.assistant import AssistantCore


class MainWindow(QMainWindow):
    """Main display window presenting the assistant face, status indicator, and Developer Mode overlay."""

    def __init__(self, assistant_core: Optional["AssistantCore"] = None):
        super().__init__()
        self._is_fullscreen = False

        # State manager & Assistant Core (lazy initialization if not provided)
        if assistant_core is None:
            from core.assistant import AssistantCore
            assistant_core = AssistantCore(parent=self)
        self.assistant = assistant_core
        self.state_manager = self.assistant.state_manager

        # Initialize developer controller
        self.dev_controller = DeveloperController(self)

        self._init_window()
        self._init_ui()
        self._connect_signals()

    def _init_window(self):
        """Configure window properties for dedicated AI device appearance."""
        self.setWindowTitle("AI Assistant")
        # Target 3.5" TFT display baseline resolution (480x320)
        self.resize(480, 320)
        self.setMinimumSize(320, 240)

        # Deep sleek dark background for OLED / TFT contrast and low power
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#0B0F17"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def _init_ui(self):
        """Build the centered user interface with status indicator and TALK dev button."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(14, 12, 14, 10)
        self.layout.setSpacing(6)

        # Main expressive face widget occupying majority of the viewport
        self.face_widget = FaceWidget(self)
        self.layout.addWidget(self.face_widget, stretch=1)

        # Bottom Bar: Status Label and TALK dev button (for normal mode)
        self.bottom_bar = QWidget(self)
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        # Subtle, clean status indicator
        self.status_label = QLabel("Ready", self.bottom_bar)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #64748B;
                font-size: 12px;
                font-weight: 500;
                letter-spacing: 1.2px;
                padding-left: 4px;
            }
        """)
        bottom_layout.addWidget(self.status_label, stretch=1)

        # TALK button (Development replacement for future physical GPIO button)
        self.talk_button = QPushButton("🎙 TALK", self.bottom_bar)
        self.talk_button.setObjectName("TalkButton")
        self.talk_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.talk_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.talk_button.setStyleSheet("""
            QPushButton#TalkButton {
                background-color: #0284C7;
                color: #FFFFFF;
                border: 1px solid #38BDF8;
                border-radius: 14px;
                padding: 5px 16px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                min-height: 28px;
            }
            QPushButton#TalkButton:hover {
                background-color: #0369A1;
                border-color: #7DD3FC;
            }
            QPushButton#TalkButton:pressed {
                background-color: #0C4A6E;
            }
        """)
        self.talk_button.clicked.connect(self.assistant.start_interaction)
        bottom_layout.addWidget(self.talk_button, stretch=0)

        self.layout.addWidget(self.bottom_bar, stretch=0)

        # Developer Mode HUD Panel (hidden by default)
        self.dev_panel = DeveloperPanel(
            self.dev_controller,
            self.assistant.generate_developer_response,
            self,
            speak_response=self.assistant.start_speaking,
            get_tts_metrics=self.assistant.get_tts_metrics,
        )
        self.dev_panel.setVisible(False)
        self.dev_panel.set_stt_engine_name(self.assistant.stt_engine_name)
        self.layout.addWidget(self.dev_panel, stretch=0)

    def _connect_signals(self):
        """Wire signals between UI components, Assistant Core, and Developer Controller."""
        # Connect state manager to UI updates
        self.state_manager.state_changed.connect(self._on_state_changed)
        self.dev_panel.ai_test_console.set_assistant_state(self.state_manager.current_state)

        # Connect Assistant Core STT, LLM, and Microphone events to UI
        self.assistant.transcription_ready.connect(self._on_transcription_ready)
        self.assistant.response_ready.connect(self._on_response_ready)
        self.assistant.telemetry_updated.connect(self._on_telemetry_updated)
        self.assistant.error_occurred.connect(self._on_assistant_error)
        self.assistant.mic.status_changed.connect(self._on_mic_status_changed)
        self.assistant.mic.speech_detected.connect(self._on_speech_detected)

        # Telemetry: Connect FaceWidget paint events to FPS counter
        self.face_widget.frame_rendered.connect(self.dev_controller.record_frame)

        # Developer controller triggers state changes and interactions
        self.dev_controller.state_change_requested.connect(self.set_state)
        self.dev_controller.interaction_requested.connect(self.assistant.start_interaction)
        self.dev_controller.mode_toggled.connect(self._on_developer_mode_toggled)

    def _on_transcription_ready(self, text: str):
        """Handle recognized transcription text."""
        self.dev_panel.set_last_transcription(text)
        if text.strip():
            self.status_label.setText(f'Heard: "{text.strip()}"')
            self.status_label.setStyleSheet("color: #38BDF8; font-size: 12px; font-weight: 600; letter-spacing: 1px;")
        else:
            self.status_label.setText("No speech detected")
            self.status_label.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 500; letter-spacing: 1px;")

    def _on_response_ready(self, text: str):
        """Handle generated response from the local LLM."""
        self.dev_panel.set_last_response(text)
        self.status_label.setText(f"AI: {text[:35]}")
        self.status_label.setStyleSheet("color: #FBBF24; font-size: 12px; font-weight: 600; letter-spacing: 1px;")

    def _on_telemetry_updated(self, rec_duration: float, stt_time: float, total_time: float):
        """Handle updated interaction performance measurements."""
        self.dev_panel.set_timing_telemetry(
            rec_duration,
            stt_time,
            total_time,
            silence_tail=self.assistant.last_silence_tail,
            time_to_speech=self.assistant.last_time_until_speech,
            capture_time=self.assistant.mic.total_capture_time,
        )

    def _on_assistant_error(self, err_msg: str):
        """Handle assistant error notifications."""
        self.status_label.setText(f"Error: {err_msg[:35]}")
        self.status_label.setStyleSheet("color: #F87171; font-size: 12px; font-weight: 600; letter-spacing: 1px;")
        self.dev_panel.set_error_message(err_msg)

    def _on_mic_status_changed(self, status: str):
        """Handle microphone recording status update."""
        duration = self.assistant.mic.recording_duration
        speech_detected = getattr(self.assistant.mic, "speech_detected_in_last_recording", False)
        self.dev_panel.set_mic_status(status, duration, speech_detected=speech_detected)

    def _on_speech_detected(self):
        """Update Developer Mode as soon as the microphone detects speech."""
        self.dev_panel.set_mic_status(
            self.assistant.mic.status,
            self.assistant.mic.recording_duration,
            speech_detected=True,
        )

    def _on_developer_mode_toggled(self, enabled: bool):
        """Show or hide the developer overlay panel."""
        self.dev_panel.setVisible(enabled)
        self.bottom_bar.setVisible(not enabled)
        if enabled:
            self.dev_panel.set_current_state(self.state_manager.current_state)
            self.dev_panel.set_stt_engine_name(self.assistant.stt_engine_name)
            self.dev_panel.set_vad_config(self.assistant.vad_silence_duration)
            self.dev_panel.set_last_transcription(self.assistant.last_transcription)
            self.dev_panel.set_last_response(self.assistant.last_response)
            self.dev_panel.set_llm_status(
                getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "model_name", getattr(self.assistant.ai, "model_name", "development")),
                getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "model_path", ""),
                str(getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "threads", "n/a")),
                str(getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "context_size", "n/a")),
                str(getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "max_tokens", "n/a")),
                getattr(getattr(self.assistant.ai, "llm", self.assistant.ai), "_load_time_seconds", 0.0),
            )
            self.dev_panel.set_timing_telemetry(
                self.assistant.last_recording_duration,
                self.assistant.last_stt_time,
                self.assistant.last_total_time,
                silence_tail=self.assistant.last_silence_tail,
                time_to_speech=self.assistant.last_time_until_speech,
                capture_time=self.assistant.mic.total_capture_time,
            )

    def set_state(self, state: AssistantState):
        """Update global assistant state via state manager."""
        self.state_manager.set_state(state)

    def _on_state_changed(self, state: AssistantState):
        """React to state changes emitted by StateManager."""
        self.face_widget.set_state(state)
        self.dev_controller.set_current_state(state)
        self.dev_panel.set_current_state(state)
        self.dev_panel.ai_test_console.set_assistant_state(state)

        # Update status label and talk button text / appearance
        if state == AssistantState.IDLE:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #64748B; font-size: 12px; font-weight: 500; letter-spacing: 1.2px;")
            self.talk_button.setText("🎙 TALK")
            self.talk_button.setEnabled(True)
            self.talk_button.setStyleSheet("""
                QPushButton#TalkButton {
                    background-color: #0284C7;
                    color: #FFFFFF;
                    border: 1px solid #38BDF8;
                    border-radius: 14px;
                    padding: 5px 16px;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    min-height: 28px;
                }
            """)

        elif state == AssistantState.LISTENING:
            self.status_label.setText("Listening...")
            self.status_label.setStyleSheet("color: #38BDF8; font-size: 12px; font-weight: 600; letter-spacing: 1.2px;")
            self.talk_button.setText("⏹ STOP")
            self.talk_button.setStyleSheet("""
                QPushButton#TalkButton {
                    background-color: #059669;
                    color: #FFFFFF;
                    border: 1px solid #34D399;
                    border-radius: 14px;
                    padding: 5px 16px;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    min-height: 28px;
                }
            """)

        elif state == AssistantState.THINKING:
            self.status_label.setText("Transcribing...")
            self.status_label.setStyleSheet("color: #FBBF24; font-size: 12px; font-weight: 600; letter-spacing: 1.2px;")
            self.talk_button.setText("Thinking")
            self.talk_button.setEnabled(False)

        elif state == AssistantState.SPEAKING:
            self.status_label.setText("Speaking...")
            self.status_label.setStyleSheet("color: #60A5FA; font-size: 12px; font-weight: 600; letter-spacing: 1.2px;")
            self.talk_button.setText("Speaking")
            self.talk_button.setEnabled(False)

        elif state == AssistantState.ERROR:
            self.status_label.setText("Error occurred")
            self.status_label.setStyleSheet("color: #F87171; font-size: 12px; font-weight: 600; letter-spacing: 1.2px;")
            self.talk_button.setText("Error")
            self.talk_button.setEnabled(False)

    def get_state(self) -> AssistantState:
        """Return current state."""
        return self.state_manager.current_state

    # =========================================================================
    # Keyboard Handling & Development Controls
    # =========================================================================

    def keyPressEvent(self, event):
        """Handle developer key events and window shortcuts."""
        key = event.key()

        # Keep developer shortcuts from consuming typed prompts in the console.
        if (
            self.dev_panel.ai_test_console.prompt_input.hasFocus()
            and key not in (Qt.Key.Key_F12, Qt.Key.Key_F11, Qt.Key.Key_Escape)
        ):
            super().keyPressEvent(event)
            return

        # First attempt handling by DeveloperController (F12, 1-5, Space, T, ESC)
        if self.dev_controller.handle_key_press(event):
            event.accept()
            return

        # F11: Toggle Fullscreen
        if key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return

        # ESC: Exit fullscreen if in fullscreen, otherwise close app
        elif key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
                self._is_fullscreen = False
            else:
                self.close()
            event.accept()
            return

        super().keyPressEvent(event)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True
