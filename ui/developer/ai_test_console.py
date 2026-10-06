"""Developer-only text console for testing the configured response engine."""

import threading
from typing import Any, Callable, Dict, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.states import AssistantState


class PromptTextEdit(QPlainTextEdit):
    """Multiline prompt editor that sends on Ctrl+Enter."""

    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AITestConsole(QFrame):
    """Submit prompts to AssistantCore's existing response engine asynchronously."""

    _generation_succeeded = Signal(str, object)
    _generation_failed = Signal(str)

    def __init__(
        self,
        generate_response: Callable[[str], Tuple[str, Dict[str, Any]]],
        parent: QWidget = None,
        speak_response: Optional[Callable[[str], bool]] = None,
        get_tts_metrics: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        super().__init__(parent)
        self._generate_response = generate_response
        self._speak_response = speak_response
        self._get_tts_metrics = get_tts_metrics
        self._is_generating = False
        self._speak_after_generation = False
        self._speech_requested = False
        self._assistant_state = AssistantState.IDLE

        self._init_ui()
        self._generation_succeeded.connect(self._show_response)
        self._generation_failed.connect(self._show_error)
        self._update_button_states()

    @property
    def is_generating(self) -> bool:
        """Return whether a console request is currently in progress."""
        return self._is_generating

    def _init_ui(self):
        self.setObjectName("AITestConsole")
        self.setStyleSheet("""
            QFrame#AITestConsole {
                background-color: rgba(15, 23, 42, 0.72);
                border: 1px solid #334155;
                border-radius: 5px;
            }
            QLabel {
                color: #CBD5E1;
                font-size: 9px;
            }
            QLabel#AITestConsoleTitle {
                color: #38BDF8;
                font-weight: 700;
                font-size: 10px;
                letter-spacing: 1px;
            }
            QLabel#AITestConsoleMetrics {
                color: #94A3B8;
                font-family: monospace;
                font-size: 8px;
            }
            QPlainTextEdit {
                background-color: #0B1220;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                selection-background-color: #0369A1;
            }
            QPushButton#AITestSendButton {
                background-color: #0369A1;
                color: #FFFFFF;
                border: 1px solid #38BDF8;
                border-radius: 4px;
                padding: 3px 12px;
                font-size: 9px;
                font-weight: 700;
            }
            QPushButton#AITestSendButton:disabled {
                background-color: #334155;
                color: #94A3B8;
                border-color: #475569;
            }
            QPushButton#AITestSpeakButton {
                background-color: #047857;
                color: #FFFFFF;
                border: 1px solid #34D399;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 9px;
                font-weight: 700;
            }
            QPushButton#AITestSpeakButton:disabled {
                background-color: #334155;
                color: #94A3B8;
                border-color: #475569;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(3)

        title = QLabel("AI TEST CONSOLE", self)
        title.setObjectName("AITestConsoleTitle")
        layout.addWidget(title)

        layout.addWidget(QLabel("Prompt:", self))
        self.prompt_input = PromptTextEdit(self)
        self.prompt_input.setObjectName("AITestPromptInput")
        self.prompt_input.setPlaceholderText("Type your prompt here...")
        self.prompt_input.setTabChangesFocus(False)
        self.prompt_input.setMinimumHeight(48)
        self.prompt_input.setMaximumHeight(66)
        layout.addWidget(self.prompt_input)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addStretch(1)
        self.send_button = QPushButton("Generate", self)
        self.send_button.setObjectName("AITestSendButton")
        self.send_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.send_button.clicked.connect(self.submit_prompt)
        action_row.addWidget(self.send_button)
        self.generate_speak_button = QPushButton("Generate & Speak", self)
        self.generate_speak_button.setObjectName("AITestSpeakButton")
        self.generate_speak_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.generate_speak_button.clicked.connect(
            lambda: self.submit_prompt(speak_after_generation=True)
        )
        action_row.addWidget(self.generate_speak_button)
        self.speak_response_button = QPushButton("Speak Response", self)
        self.speak_response_button.setObjectName("AITestSpeakButton")
        self.speak_response_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speak_response_button.clicked.connect(self.speak_current_response)
        action_row.addWidget(self.speak_response_button)
        self.generation_status = QLabel("", self)
        action_row.addWidget(self.generation_status)
        layout.addLayout(action_row)

        layout.addWidget(QLabel("Response:", self))
        self.response_output = QPlainTextEdit(self)
        self.response_output.setObjectName("AITestResponseOutput")
        self.response_output.setReadOnly(True)
        self.response_output.setPlaceholderText("TinyLlama response appears here...")
        self.response_output.setMinimumHeight(42)
        self.response_output.setMaximumHeight(64)
        layout.addWidget(self.response_output)

        self.metrics_label = QLabel(
            "Model: -- | Generation time: -- | Generated tokens: -- | Tokens/sec: --",
            self,
        )
        self.metrics_label.setObjectName("AITestConsoleMetrics")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

        self.tts_metrics_label = QLabel("TTS metrics: unavailable", self)
        self.tts_metrics_label.setObjectName("AITestConsoleMetrics")
        self.tts_metrics_label.setWordWrap(True)
        layout.addWidget(self.tts_metrics_label)

        self.prompt_input.send_requested.connect(self.submit_prompt)

    def submit_prompt(self, speak_after_generation: bool = False) -> bool:
        """Start one background generation, rejecting empty or overlapping prompts."""
        prompt = self.prompt_input.toPlainText()
        if not prompt.strip():
            self.generation_status.setText("Enter a prompt to generate.")
            return False
        if not self._can_start_generation():
            return False

        self._is_generating = True
        self._speak_after_generation = speak_after_generation
        self.generation_status.setText("Generating...")
        self.response_output.setPlainText("Generating...")
        self.metrics_label.setText("Model: -- | Generation time: -- | Generated tokens: -- | Tokens/sec: --")
        self._update_button_states()
        threading.Thread(
            target=self._generation_worker,
            args=(prompt,),
            daemon=True,
        ).start()
        return True

    def _generation_worker(self, prompt: str):
        try:
            response, metrics = self._generate_response(prompt)
            self._generation_succeeded.emit(response, metrics)
        except Exception as exc:
            self._generation_failed.emit(str(exc))

    def _show_response(self, response: str, metrics: Dict[str, Any]):
        engine_status = metrics.get("engine_status")
        status_text = f" | Status: {engine_status}" if engine_status else ""
        self.response_output.setPlainText(response)
        self.metrics_label.setText(
            "Model: {model} | Generation time: {duration} s | "
            "Generated tokens: {tokens} | Tokens/sec: {rate}{status}".format(
                model=metrics.get("model_name", "--"),
                duration=self._metric_value(metrics, "generation_time_seconds", "elapsed_seconds"),
                tokens=metrics.get("generated_tokens", "--"),
                rate=self._metric_value(metrics, "tokens_per_second"),
                status=status_text,
            )
        )
        should_speak = self._speak_after_generation
        self._speak_after_generation = False
        self._finish_generation()
        if should_speak:
            self._speak_text(response)

    def _show_error(self, error: str):
        self.response_output.setPlainText(
            f"ERROR\n\nTinyLlama generation failed:\n{error}"
        )
        self.metrics_label.setText("Model: TinyLlama | Generation failed")
        self._finish_generation()

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                try:
                    return f"{float(value):.3f}" if key != "tokens_per_second" else f"{float(value):.2f}"
                except (TypeError, ValueError):
                    return str(value)
        return "--"

    def _finish_generation(self):
        self._is_generating = False
        self._speak_after_generation = False
        self._update_button_states()
        self.generation_status.clear()

    def speak_current_response(self) -> bool:
        """Speak the text currently displayed without generating another response."""
        response = self.response_output.toPlainText().strip()
        if not response or response == "Generating..." or response.startswith("ERROR"):
            self.generation_status.setText("There is no response to speak.")
            return False
        if self._is_generating or not self._can_speak():
            return False
        return self._speak_text(response)

    def _speak_text(self, text: str) -> bool:
        if not text.strip():
            self.generation_status.setText("There is no response to speak.")
            return False
        if self._speak_response is None:
            self.generation_status.setText("Speech is unavailable.")
            return False
        try:
            accepted = self._speak_response(text)
        except Exception as exc:
            self.generation_status.setText(f"Speech request failed: {exc}")
            return False
        if accepted is False:
            self.generation_status.setText("Speech is already active.")
            return False
        self._speech_requested = True
        self.generation_status.setText("Speaking...")
        self._update_button_states()
        return True

    def set_assistant_state(self, state: AssistantState):
        """Update control availability and TTS metrics from AssistantCore state."""
        self._assistant_state = state
        if state == AssistantState.IDLE:
            self._speech_requested = False
            self.generation_status.clear()
        self._update_button_states()
        self.update_tts_metrics()

    def update_tts_metrics(self):
        """Display the active TTS engine metrics that are currently available."""
        metrics = self._get_tts_metrics() if self._get_tts_metrics else {}
        if not metrics:
            self.tts_metrics_label.setText("TTS metrics: unavailable")
            return

        model = metrics.get("voice") or metrics.get("model_name")
        fields = []
        if model:
            fields.append(f"Voice/model: {model}")
        for key, label in (
            ("synthesis_time_seconds", "Synthesis"),
            ("audio_duration_seconds", "Audio"),
            ("real_time_factor", "RTF"),
        ):
            value = metrics.get(key)
            if value is not None:
                try:
                    formatted = f"{float(value):.3f}"
                    unit = "" if key == "real_time_factor" else "s"
                    fields.append(f"{label}: {formatted}{unit}")
                except (TypeError, ValueError):
                    fields.append(f"{label}: {value}")
        self.tts_metrics_label.setText("TTS | " + " | ".join(fields) if fields else "TTS metrics: unavailable")

    def _can_start_generation(self) -> bool:
        return (
            not self._is_generating
            and not self._speech_requested
            and self._assistant_state not in (AssistantState.SPEAKING, AssistantState.ERROR)
        )

    def _can_speak(self) -> bool:
        return (
            self._speak_response is not None
            and not self._speech_requested
            and self._assistant_state not in (AssistantState.SPEAKING, AssistantState.ERROR)
        )

    def _update_button_states(self):
        can_generate = self._can_start_generation()
        can_speak = not self._is_generating and self._can_speak()
        self.send_button.setEnabled(can_generate)
        self.generate_speak_button.setEnabled(can_generate and self._speak_response is not None)
        self.speak_response_button.setEnabled(can_speak)
