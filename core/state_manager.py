"""State Manager for AI Assistant state lifecycle and notifications."""

import logging
from PySide6.QtCore import QObject, Signal
from ui.states import AssistantState

logger = logging.getLogger("pi_assistant.state_manager")


class StateManager(QObject):
    """Central state manager storing current assistant state and notifying listeners."""

    state_changed = Signal(AssistantState)

    def __init__(self, initial_state: AssistantState = AssistantState.IDLE, parent=None):
        super().__init__(parent)
        self._current_state = initial_state

    @property
    def current_state(self) -> AssistantState:
        """Return the current assistant state."""
        return self._current_state

    def get_state(self) -> AssistantState:
        """Return the current assistant state."""
        return self._current_state

    def set_state(self, new_state: AssistantState):
        """Transition to a new state and emit notification signal."""
        if self._current_state == new_state:
            return

        old_state = self._current_state
        self._current_state = new_state
        logger.info("[StateManager] State changed: %s -> %s", old_state.name, new_state.name)
        self.state_changed.emit(self._current_state)
