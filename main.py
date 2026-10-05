"""Entry point for Raspberry Pi AI Assistant.

Runs the PySide6 UI and Assistant Core with PC voice development mode.

Run with:
    python main.py

Controls during development:
    TALK (Button) : Trigger PC microphone recording & AI voice response
    T / Enter     : Keyboard shortcut to trigger TALK
    F12           : Toggle Developer Mode HUD overlay
    Ctrl+Enter    : Send a multiline prompt in Developer Mode
    1 - 5         : Set states manually (1=IDLE, 2=LISTENING, 3=THINKING, 4=SPEAKING, 5=ERROR)
    SPACE         : Toggle between IDLE and SPEAKING
    F11           : Toggle fullscreen
    ESC           : Close Developer Mode / Exit fullscreen / Close application
"""

import logging
import sys
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from core.assistant import AssistantCore
from ui.main_window import MainWindow

# Configure clean structured logging for assistant interactions
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pi_assistant")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Pi AI Assistant")

    # Set default clean system font
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Initialize Assistant Core and Main Window
    assistant_core = AssistantCore()
    window = MainWindow(assistant_core=assistant_core)
    window.show()

    logger.info("[App] Raspberry Pi AI Assistant started in PC Development Mode.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
