"""Developer controller managing Developer Mode state, FPS telemetry, and shortcuts."""

import time
from PySide6.QtCore import QObject, QTimer, Qt, Signal

from ui.states import AssistantState


class DeveloperController(QObject):
    """Coordinates Developer Mode activation, keyboard shortcuts, and state changes.

    Acts as the testing bridge between dev inputs and UI state changes,
    ensuring identical state dispatching as the Assistant Core.
    """

    # Signals
    mode_toggled = Signal(bool)
    state_change_requested = Signal(AssistantState)
    interaction_requested = Signal()
    fps_updated = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_enabled = False
        self._current_state = AssistantState.IDLE

        # FPS calculation metrics
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._current_fps = 0.0

        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)  # Update FPS telemetry every second
        self._fps_timer.timeout.connect(self._calculate_fps)
        self._fps_timer.start()

    @property
    def is_enabled(self) -> bool:
        """Return whether Developer Mode is currently active."""
        return self._is_enabled

    def toggle_mode(self):
        """Toggle Developer Mode between active and inactive."""
        self.set_mode(not self._is_enabled)

    def set_mode(self, enabled: bool):
        """Explicitly enable or disable Developer Mode."""
        if self._is_enabled != enabled:
            self._is_enabled = enabled
            self.mode_toggled.emit(self._is_enabled)

    def set_current_state(self, state: AssistantState):
        """Update controller's knowledge of the current state."""
        self._current_state = state

    def request_state(self, state: AssistantState):
        """Request a state transition to the state manager / main window."""
        self.state_change_requested.emit(state)

    def request_interaction(self):
        """Request a full voice interaction cycle (same as TALK button)."""
        self.interaction_requested.emit()

    def record_frame(self):
        """Record a rendered frame for FPS calculation."""
        self._frame_count += 1

    def _calculate_fps(self):
        """Calculate and emit the current frame rate."""
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed > 0:
            self._current_fps = self._frame_count / elapsed
            self.fps_updated.emit(self._current_fps)
        self._frame_count = 0
        self._last_fps_time = now

    def handle_key_press(self, event) -> bool:
        """Handle developer key shortcuts.

        Supported shortcuts:
            - F12   : Toggle Developer Mode
            - 1     : Set state to IDLE
            - 2     : Set state to LISTENING
            - 3     : Set state to THINKING
            - 4     : Set state to SPEAKING
            - 5     : Set state to ERROR
            - T     : Trigger voice interaction (TALK)
            - SPACE : Toggle between IDLE and SPEAKING
            - ESC   : Close Developer Mode if open

        Returns:
            bool: True if the event was handled by Developer Controller, False otherwise.
        """
        key = event.key()

        # F12: Toggle Developer Mode (works globally)
        if key == Qt.Key.Key_F12:
            self.toggle_mode()
            return True

        # T or Return/Enter: Trigger TALK interaction
        if key in (Qt.Key.Key_T, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.request_interaction()
            return True

        # SPACE: Toggle state between IDLE and SPEAKING
        if key == Qt.Key.Key_Space:
            new_state = (
                AssistantState.SPEAKING
                if self._current_state == AssistantState.IDLE
                else AssistantState.IDLE
            )
            self.request_state(new_state)
            return True

        # Number shortcuts when in Developer Mode
        if self._is_enabled:
            state_map = {
                Qt.Key.Key_1: AssistantState.IDLE,
                Qt.Key.Key_2: AssistantState.LISTENING,
                Qt.Key.Key_3: AssistantState.THINKING,
                Qt.Key.Key_4: AssistantState.SPEAKING,
                Qt.Key.Key_5: AssistantState.ERROR,
            }
            if key in state_map:
                self.request_state(state_map[key])
                return True

            # ESC: Close Developer Mode if open
            if key == Qt.Key.Key_Escape:
                self.set_mode(False)
                return True

        return False
