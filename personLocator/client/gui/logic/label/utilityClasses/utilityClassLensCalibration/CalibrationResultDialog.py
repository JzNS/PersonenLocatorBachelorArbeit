from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QScrollArea, QWidget, QGridLayout, QVBoxLayout, QLabel, QDialog, QPushButton
import cv2
import numpy as np

class CalibrationResultDialog(QDialog):
    """Gibt eine übersichtliche Darstellung der Original- und
    Entzerrten Bilder mit Fehlerangaben, um die Kalibrierungsqualität zu überprüfen."""
    def __init__(self, images, mtx, dist, errors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Überprüfung: Gefundene Linsen-Verzerrung")
        self.resize(1300, 800)

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)

        row, col = 0, 0
        max_cols = 3

        for i, img_bgr in enumerate(images):
            undistorted_bgr = cv2.undistort(img_bgr, mtx, dist)
            combined_bgr = np.vstack((img_bgr, undistorted_bgr))

            h, w = combined_bgr.shape[:2]
            scale = 400 / w
            new_w, new_h = int(w * scale), int(h * scale)
            combined_resized = cv2.resize(combined_bgr, (new_w, new_h))

            combined_rgb = cv2.cvtColor(combined_resized, cv2.COLOR_BGR2RGB)
            qimg = QImage(combined_rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            error_val = errors[i] if i < len(errors) else 0.0
            title_text = f"Bild {i + 1} | Fehler: {error_val:.2f} Pixel\nOrig (Oben) -> Entzerrt (Unten)"

            bg_color = "#1a331a" if error_val < 1.0 else ("#4c3300" if error_val < 2.5 else "#330000")

            title = QLabel(title_text)
            title.setStyleSheet(f"font-weight: bold; color: white; background: {bg_color}; padding: 5px;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            box = QVBoxLayout()
            box.addWidget(title)
            box.addWidget(lbl)

            wrapper = QWidget()
            wrapper.setLayout(box)
            wrapper.setStyleSheet("border: 1px solid #555; background: #222;")

            grid.addWidget(wrapper, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        scroll.setWidget(container)
        layout.addWidget(scroll)

        btn_ok = QPushButton("Alles sieht super aus! Speichern & Schließen")
        btn_ok.setFixedHeight(60)
        btn_ok.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 16px;")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)