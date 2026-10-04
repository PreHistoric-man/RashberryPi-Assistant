"""FaceWidget for Raspberry Pi AI Assistant.

Renders a minimal, friendly facial UI using QPainter and smooth animations.
Supports 5 visual states:
- IDLE: Smiling face with natural blinking.
- LISTENING: Perked attentive eyes and focused listening expression.
- THINKING: Smooth pondering eye animation.
- SPEAKING: Animated talking mouth cycling through speech visemes.
- ERROR: Expressive error visual state with recovery.
"""

import math
import random
from PySide6.QtCore import QPointF, QRectF, QTime, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .states import AssistantState


class FaceWidget(QWidget):
    """Custom widget rendering the assistant's expressive face using vector QPainter."""

    frame_rendered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = AssistantState.IDLE

        # Visual styling
        self._face_color = QColor("#F1F5F9")      # Crisp soft white
        self._error_color = QColor("#F87171")     # Soft coral red for error
        self._listen_color = QColor("#38BDF8")    # Vibrant cyan for listening

        # Eye blink animation state
        self._blink_progress = 1.0  # 1.0 = fully open, 0.0 = fully closed
        self._is_blinking = False
        self._blink_direction = -1  # -1 closing, 1 opening
        self._blink_timer = QTimer(self)
        self._blink_timer.setSingleShot(True)
        self._blink_timer.timeout.connect(self._start_blink)

        self._blink_anim_timer = QTimer(self)
        self._blink_anim_timer.setInterval(16)  # ~60fps for smooth blink
        self._blink_anim_timer.timeout.connect(self._update_blink_animation)

        # Thinking animation state (smooth eye gaze shift)
        self._thinking_angle = 0.0
        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(25)  # 40fps for smooth gaze swing
        self._thinking_timer.timeout.connect(self._update_thinking_animation)

        # Listening pulse animation state
        self._listening_pulse = 0.0
        self._listening_timer = QTimer(self)
        self._listening_timer.setInterval(30)
        self._listening_timer.timeout.connect(self._update_listening_animation)

        # Speaking animation state
        # Speech viseme target shapes: (mouth_type, width_ratio, height_ratio, curve_ratio)
        # types: 0 = arc smile (◡), 1 = open circle (○), 2 = wide bar/pill (▬), 3 = open smile (ᴗ)
        self._viseme_targets = [
            (0, 1.0, 0.35, 1.0),   # ◡ smile arc
            (1, 0.55, 0.70, 0.1),  # ○ open circle
            (2, 1.15, 0.28, 0.0),  # ▬ wide bar / pill
            (1, 0.75, 0.85, 0.2),  # ○ wide round
            (0, 0.90, 0.30, 0.9),  # ◡ gentle smile
            (3, 0.80, 0.55, 0.7),  # ᴗ open smile
            (2, 0.95, 0.32, 0.0),  # ▬ neutral bar
        ]
        self._current_viseme_idx = 0

        # Current interpolated mouth parameters
        self._cur_mouth_type = 0
        self._cur_width_ratio = 1.0
        self._cur_height_ratio = 0.35
        self._cur_curve_ratio = 1.0

        self._target_mouth_type = 0
        self._target_width_ratio = 1.0
        self._target_height_ratio = 0.35
        self._target_curve_ratio = 1.0

        self._speech_timer = QTimer(self)
        self._speech_timer.setInterval(20)  # smooth frame rate for speech morphing
        self._speech_timer.timeout.connect(self._update_speaking_animation)

        self._viseme_change_timer = QTimer(self)
        self._viseme_change_timer.setInterval(120)  # cadence of speech phonemes
        self._viseme_change_timer.timeout.connect(self._next_viseme)

        # Start initial blink schedule
        self._schedule_next_blink()

    def set_state(self, state: AssistantState):
        """Update the assistant state and adjust facial animations accordingly."""
        if self._state == state:
            return

        self._state = state

        # Manage state-specific timers
        if self._state == AssistantState.SPEAKING:
            self._thinking_timer.stop()
            self._listening_timer.stop()
            self._viseme_change_timer.start()
            self._speech_timer.start()
        elif self._state == AssistantState.THINKING:
            self._viseme_change_timer.stop()
            self._speech_timer.stop()
            self._listening_timer.stop()
            self._thinking_timer.start()
        elif self._state == AssistantState.LISTENING:
            self._viseme_change_timer.stop()
            self._speech_timer.stop()
            self._thinking_timer.stop()
            self._listening_timer.start()
        elif self._state == AssistantState.IDLE:
            self._thinking_timer.stop()
            self._listening_timer.stop()
            self._viseme_change_timer.stop()
            self._speech_timer.stop()
            # Reset mouth smoothly to idle smile
            self._cur_mouth_type = 0
            self._cur_width_ratio = 1.0
            self._cur_height_ratio = 0.35
            self._cur_curve_ratio = 1.0
            self._target_mouth_type = 0
            self._target_width_ratio = 1.0
            self._target_height_ratio = 0.35
            self._target_curve_ratio = 1.0
        elif self._state == AssistantState.ERROR:
            self._thinking_timer.stop()
            self._listening_timer.stop()
            self._viseme_change_timer.stop()
            self._speech_timer.stop()

        self.update()

    def get_state(self) -> AssistantState:
        """Get the current assistant state."""
        return self._state

    # --- Blinking Mechanism ---

    def _schedule_next_blink(self):
        """Schedule the next random blink interval (every 2.8 - 5.5 seconds)."""
        interval_ms = random.randint(2800, 5500)
        self._blink_timer.start(interval_ms)

    def _start_blink(self):
        """Trigger an eye blink."""
        if not self._is_blinking and self._state != AssistantState.ERROR:
            self._is_blinking = True
            self._blink_direction = -1
            self._blink_progress = 1.0
            self._blink_anim_timer.start()
        else:
            self._schedule_next_blink()

    def _update_blink_animation(self):
        """Animate the closing and opening of the eyelids."""
        step = 0.16
        if self._blink_direction == -1:
            self._blink_progress -= step
            if self._blink_progress <= 0.05:
                self._blink_progress = 0.05
                self._blink_direction = 1  # Start reopening
        else:
            self._blink_progress += step
            if self._blink_progress >= 1.0:
                self._blink_progress = 1.0
                self._is_blinking = False
                self._blink_anim_timer.stop()
                self._schedule_next_blink()

        self.update()

    # --- Thinking Animation Mechanism ---

    def _update_thinking_animation(self):
        """Oscillate gaze angle smoothly while thinking."""
        self._thinking_angle += 0.08
        if self._thinking_angle > 2 * math.pi:
            self._thinking_angle -= 2 * math.pi
        self.update()

    # --- Listening Pulse Animation Mechanism ---

    def _update_listening_animation(self):
        """Subtle energetic pulse while actively listening."""
        self._listening_pulse += 0.07
        if self._listening_pulse > 2 * math.pi:
            self._listening_pulse -= 2 * math.pi
        self.update()

    # --- Speaking Animation Mechanism ---

    def _next_viseme(self):
        """Pick the next target mouth shape during speech."""
        self._current_viseme_idx = (self._current_viseme_idx + 1) % len(self._viseme_targets)
        if random.random() < 0.3:
            self._current_viseme_idx = random.randint(0, len(self._viseme_targets) - 1)

        target = self._viseme_targets[self._current_viseme_idx]
        self._target_mouth_type = target[0]
        self._target_width_ratio = target[1]
        self._target_height_ratio = target[2]
        self._target_curve_ratio = target[3]

    def _update_speaking_animation(self):
        """Smoothly interpolate towards the target mouth shape."""
        lerp = 0.28
        self._cur_width_ratio += (self._target_width_ratio - self._cur_width_ratio) * lerp
        self._cur_height_ratio += (self._target_height_ratio - self._cur_height_ratio) * lerp
        self._cur_curve_ratio += (self._target_curve_ratio - self._cur_curve_ratio) * lerp

        if abs(self._target_mouth_type - self._cur_mouth_type) > 0.01:
            self._cur_mouth_type = self._target_mouth_type

        self.update()

    # --- Drawing Logic ---

    def paintEvent(self, event):
        """Render the face centered within the widget boundaries."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        cx = w / 2.0
        cy = h / 2.0

        # Scale factor tuned for standard 480x320 TFT baseline
        scale = min(w / 480.0, h / 320.0)
        scale = max(scale, 0.4)

        # Baseline facial dimensions
        eye_spacing = 72.0 * scale       # distance from center to each eye
        eye_y = cy - 24.0 * scale        # eyes vertical position
        eye_radius_x = 15.0 * scale      # eye width
        eye_radius_y = 17.0 * scale      # eye height
        stroke_width = max(5.0 * scale, 3.0)

        mouth_center_y = cy + 34.0 * scale
        base_mouth_w = 64.0 * scale
        base_mouth_h = 32.0 * scale

        # State color selection
        draw_color = self._face_color
        if self._state == AssistantState.ERROR:
            draw_color = self._error_color
        elif self._state == AssistantState.LISTENING:
            draw_color = self._face_color

        # Draw Eyes
        self._draw_eyes(painter, cx, eye_y, eye_spacing, eye_radius_x, eye_radius_y, stroke_width, scale, draw_color)

        # Draw Mouth
        self._draw_mouth(painter, cx, mouth_center_y, base_mouth_w, base_mouth_h, stroke_width, draw_color)

        self.frame_rendered.emit()

    def _draw_eyes(self, painter: QPainter, cx: float, eye_y: float, spacing: float, rx: float, ry: float, stroke_w: float, scale: float, color: QColor):
        """Draw both eyes with state-specific expression and blinking."""
        left_x = cx - spacing
        right_x = cx + spacing

        if self._state == AssistantState.ERROR:
            # Error State: Draw > < or x x squint expression
            pen = QPen(color, stroke_w * 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            d = rx * 0.9
            # Left eye: >
            painter.drawLine(QPointF(left_x - d, eye_y - d), QPointF(left_x + d * 0.6, eye_y))
            painter.drawLine(QPointF(left_x + d * 0.6, eye_y), QPointF(left_x - d, eye_y + d))
            # Right eye: <
            painter.drawLine(QPointF(right_x + d, eye_y - d), QPointF(right_x - d * 0.6, eye_y))
            painter.drawLine(QPointF(right_x - d * 0.6, eye_y), QPointF(right_x + d, eye_y + d))
            return

        if self._state == AssistantState.THINKING:
            # Thinking State: Eyes gently shift gaze horizontally and slightly up
            gaze_offset_x = math.sin(self._thinking_angle) * (8.0 * scale)
            gaze_offset_y = -abs(math.cos(self._thinking_angle)) * (3.0 * scale)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(left_x + gaze_offset_x - rx, eye_y + gaze_offset_y - ry, rx * 2.0, ry * 2.0))
            painter.drawEllipse(QRectF(right_x + gaze_offset_x - rx, eye_y + gaze_offset_y - ry, rx * 2.0, ry * 2.0))
            return

        if self._state == AssistantState.LISTENING:
            # Listening State: Attentive wide eyes with subtle alive breathing
            pulse_extra = math.sin(self._listening_pulse) * (2.0 * scale)
            cur_rx = rx + pulse_extra * 0.5
            cur_ry = ry + pulse_extra * 0.5

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(left_x - cur_rx, eye_y - cur_ry, cur_rx * 2.0, cur_ry * 2.0))
            painter.drawEllipse(QRectF(right_x - cur_rx, eye_y - cur_ry, cur_rx * 2.0, cur_ry * 2.0))
            return

        # IDLE / SPEAKING State: Normal eyes with blink handling
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))

        current_ry = max(ry * self._blink_progress, stroke_w * 0.5)

        if self._blink_progress <= 0.12:
            # Closed blink slit
            pen = QPen(color, stroke_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            slit_w = rx * 1.1
            painter.drawLine(QPointF(left_x - slit_w, eye_y), QPointF(left_x + slit_w, eye_y))
            painter.drawLine(QPointF(right_x - slit_w, eye_y), QPointF(right_x + slit_w, eye_y))
        else:
            # Open oval eyes
            left_eye_rect = QRectF(left_x - rx, eye_y - current_ry, rx * 2.0, current_ry * 2.0)
            right_eye_rect = QRectF(right_x - rx, eye_y - current_ry, rx * 2.0, current_ry * 2.0)
            painter.drawEllipse(left_eye_rect)
            painter.drawEllipse(right_eye_rect)

    def _draw_mouth(self, painter: QPainter, cx: float, cy: float, base_w: float, base_h: float, stroke_w: float, color: QColor):
        """Draw mouth according to current state and viseme parameters."""
        pen = QPen(color, stroke_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        if self._state == AssistantState.ERROR:
            # Error State: Flat or slight concern mouth
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mw = base_w * 0.7
            path = QPainterPath()
            path.moveTo(QPointF(cx - mw * 0.5, cy + base_h * 0.2))
            path.quadTo(QPointF(cx, cy - base_h * 0.3), QPointF(cx + mw * 0.5, cy + base_h * 0.2))
            painter.drawPath(path)
            return

        if self._state == AssistantState.THINKING:
            # Thinking State: Gentle curious dot or small pleasant wave
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mw = base_w * 0.6
            path = QPainterPath()
            path.moveTo(QPointF(cx - mw * 0.5, cy))
            path.quadTo(QPointF(cx - mw * 0.2, cy - base_h * 0.2), QPointF(cx, cy))
            path.quadTo(QPointF(cx + mw * 0.2, cy + base_h * 0.2), QPointF(cx + mw * 0.5, cy))
            painter.drawPath(path)
            return

        if self._state == AssistantState.LISTENING:
            # Listening State: Attentive open smile
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mw = base_w * 0.85
            path = QPainterPath()
            start_pt = QPointF(cx - mw * 0.5, cy - base_h * 0.3)
            end_pt = QPointF(cx + mw * 0.5, cy - base_h * 0.3)
            ctrl_pt = QPointF(cx, cy + base_h * 0.85)
            path.moveTo(start_pt)
            path.quadTo(ctrl_pt, end_pt)
            painter.drawPath(path)
            return

        if self._state == AssistantState.IDLE or self._cur_mouth_type == 0:
            # ◡ Simple smile arc (quadratic Bezier curve)
            mw = base_w * self._cur_width_ratio
            mh = base_h * self._cur_height_ratio
            painter.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            start_pt = QPointF(cx - mw * 0.5, cy - mh * 0.4)
            end_pt = QPointF(cx + mw * 0.5, cy - mh * 0.4)
            ctrl_pt = QPointF(cx, cy + mh * 0.9 * self._cur_curve_ratio)

            path.moveTo(start_pt)
            path.quadTo(ctrl_pt, end_pt)
            painter.drawPath(path)

        elif self._cur_mouth_type == 1:
            # ○ Open round mouth (ellipse)
            mw = base_w * self._cur_width_ratio
            mh = base_h * self._cur_height_ratio
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(cx - mw * 0.35, cy - mh * 0.5, mw * 0.7, mh)
            painter.drawEllipse(rect)

        elif self._cur_mouth_type == 2:
            # ▬ Wide horizontal bar / pill shape
            mw = base_w * self._cur_width_ratio
            painter.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            start_pt = QPointF(cx - mw * 0.48, cy)
            end_pt = QPointF(cx + mw * 0.48, cy)
            path.moveTo(start_pt)
            path.lineTo(end_pt)
            painter.drawPath(path)

        elif self._cur_mouth_type == 3:
            # ᴗ Open smile
            mw = base_w * self._cur_width_ratio
            mh = base_h * self._cur_height_ratio
            painter.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            start_pt = QPointF(cx - mw * 0.45, cy - mh * 0.3)
            end_pt = QPointF(cx + mw * 0.45, cy - mh * 0.3)
            ctrl_pt = QPointF(cx, cy + mh * 0.8)

            path.moveTo(start_pt)
            path.quadTo(ctrl_pt, end_pt)
            path.lineTo(start_pt)
            painter.drawPath(path)
