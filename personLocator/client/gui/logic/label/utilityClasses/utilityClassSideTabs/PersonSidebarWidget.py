from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette
import cv2
import numpy as np


class PersonSidebarWidget(QWidget):
    """
    Verwaltet die Liste der erkannten Personen.
    NEU: Klickbar zur Auswahl einer Person.
    """
    person_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)

        self.layout_list = QVBoxLayout(self)
        self.layout_list.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout_list.setContentsMargins(0, 0, 0, 0)

        self.person_widgets = {}
        self.selected_id = None

    def set_selected_id(self, p_id):
        """Setzt die aktive ID und aktualisiert das Styling (Rahmen)."""
        self.selected_id = p_id
        self._refresh_styles()

    def update_persons(self, persons: list):
        """Aktualisiert die Anzeige der Personen
        basierend auf der übergebenen Liste von Dictionaries."""
        active_ids = [p['id'] for p in persons]

        for p_id in list(self.person_widgets.keys()):
            if p_id not in active_ids:
                widget = self.person_widgets.pop(p_id)
                self.layout_list.removeWidget(widget)
                widget.deleteLater()

        target_size = QSize(220, 200)

        for p in persons:
            p_id = p['id']
            thumb = p.get('thumbnail')
            stable_h = p.get('stable_height') or 0
            is_skel = p.get('is_skeleton', False)

            if p_id not in self.person_widgets:
                self.person_widgets[p_id] = self._create_person_item(target_size, p_id)
                self.layout_list.addWidget(self.person_widgets[p_id])

            widget = self.person_widgets[p_id]
            self._update_item_content(widget, thumb, stable_h, is_skel, target_size)

        self._refresh_styles()

    def _refresh_styles(self):
        """Zeichnet Rahmen um das ausgewählte Widget."""
        for p_id, widget in self.person_widgets.items():
            if p_id == self.selected_id:
                widget.lbl_img.setStyleSheet("border: 3px solid #00AAFF; background-color: #222;")
            else:
                widget.lbl_img.setStyleSheet("border: 1px solid #333; background-color: #111; color: #555;")

    def _create_person_item(self, size: QSize, p_id: int) -> QWidget:
        """Erstellt das Widget für eine einzelne Person mit Bild und Text."""
        container = ClickableContainer(p_id)
        container.clicked.connect(self.person_clicked.emit)

        container.setFixedWidth(220)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 5)
        vbox.setSpacing(0)

        lbl_img = QLabel()
        lbl_img.setFixedSize(size)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_txt = QLabel("---")
        lbl_txt.setFixedHeight(30)
        lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = lbl_txt.font();
        font.setPointSize(12);
        font.setBold(True)
        lbl_txt.setFont(font)

        vbox.addWidget(lbl_img)
        vbox.addWidget(lbl_txt)

        container.lbl_img = lbl_img
        container.lbl_txt = lbl_txt
        return container

    def _update_item_content(self, widget, thumb, height, is_skel, size):
        """Aktualisiert das Bild und den Text eines Personen-Widgets."""
        if thumb is not None and thumb.size > 0:
            try:
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                widget.lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(
                    size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
                ))
            except Exception:
                widget.lbl_img.clear()
                widget.lbl_img.setText("Error")
        else:
            widget.lbl_img.clear()
            widget.lbl_img.setText("Kein Bild\n(Performance)")

        if height is not None and height > 0:
            widget.lbl_txt.setText(f"{height:.0f} cm")
            color = "#0000AA" if is_skel else "#006400"
            widget.lbl_txt.setStyleSheet(f"background-color: {color}; color: white; border-radius: 4px;")
        else:
            widget.lbl_txt.setText("Kalibrierung...")
            widget.lbl_txt.setStyleSheet("color: gray;")


# --- Hilfsklasse für Klicks ---
class ClickableContainer(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, p_id):
        super().__init__()
        self.p_id = p_id

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.p_id)
        super().mousePressEvent(event)