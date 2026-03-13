from typing import Optional
import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent


class Calibration3DView(QLabel):
    rotation_requested = pyqtSignal(int, int)
    zoom_requested = pyqtSignal(int)

    STYLE_SHEET = "border: 2px solid #444; background-color: #050505; color: #888;"
    MIN_SIZE = (400, 300)

    def __init__(self, title: str = "3D Ansicht"):
        super().__init__(title)
        self._last_mouse_pos: Optional[QPoint] = None
        self._setup_ui()
        self.setScaledContents(False)

    def _setup_ui(self) -> None:
        """Initialisiert das visuelle Erscheinungsbild."""
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self.STYLE_SHEET)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(*self.MIN_SIZE)


    def update_frame(self, image: QImage) -> None:
        """
        Zeigt das Bild an.
        Nutzt SmoothTransformation, um Texte und feine OpenCV-Linien beim
        Resizing in der GUI gestochen scharf und ohne Artefakte darzustellen.
        """
        # 1. Performance-Check
        if image.isNull() or not self.isVisible() or self.width() < 10 or self.height() < 10:
            return

        # 2. Zielgröße berechnen
        target_size = self.size()

        # 3. Skalieren + Scharfe Darstellung
        scaled_img = image.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 4. In Pixmap wandeln und setzen
        self.setPixmap(QPixmap.fromImage(scaled_img))

    # --- INTERAKTION (Maus) ---

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        """Berechnet die Maus-Bewegung und sendet Rotations-Deltas."""
        if self._last_mouse_pos is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_mouse_pos

            self.rotation_requested.emit(delta.x(), delta.y())
            self._last_mouse_pos = current_pos

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Berechnet die Zoom-Stufen basierend
        auf der Mausradbewegung und sendet sie als Signal."""
        delta = event.angleDelta().y()
        steps = delta // 120
        if steps != 0:
            self.zoom_requested.emit(steps)
        event.accept()