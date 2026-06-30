import logging
import time

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QSizePolicy
)
from PyQt6 import QtWidgets as qtw
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSlot

from client.gui.logic.label.utilityClasses.utilityClassLensCalibration.CalibrationResultDialog import \
    CalibrationResultDialog
from client.gui.logic.label.utilityClasses.utilityClassLensCalibration.LensCalibrationLogic import \
    LensCalibrationLogic
from client.utils.ConfigManager import ConfigManager

import numpy as np
import cv2


class LensCalibrationWindow(QDialog):
    def __init__(self, client_name: str, parent=None):
        super().__init__(parent)
        self.client_name = client_name

        self.logic = LensCalibrationLogic((8, 6), 25.0)
        self.current_corners = None
        self._is_processing = False

        self._setup_ui()
        self.setWindowTitle("Linsen-Kalibrierung (Schachbrett)")
        self.resize(1280, 720)

        ConfigManager.set_edit_mode(True)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_camera = QLabel("Warte auf Kamera...")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background: black;")
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl_camera, stretch=1)

        self.lbl_info = QLabel("Halte das Schachbrett in die Kamera. Mindestens 15 Bilder benötigt!")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: yellow;")
        layout.addWidget(self.lbl_info)

        btn_layout = QHBoxLayout()

        self.btn_capture = QPushButton("📸 Bild aufnehmen (0/15)")
        self.btn_capture.setFixedHeight(50)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self._on_capture_clicked)

        self.btn_calc = QPushButton("⚙️ Verzerrung berechnen")
        self.btn_calc.setFixedHeight(50)
        self.btn_calc.clicked.connect(self._on_calc_clicked)

        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_calc)
        layout.addLayout(btn_layout)

    @pyqtSlot(QImage)
    def update_frame(self, frame_qimage: QImage):
        if frame_qimage.isNull():
            self._unlock_worker()
            return

        try:
            frame_rgb = frame_qimage.convertToFormat(QImage.Format.Format_RGB888)
            width, height = frame_rgb.width(), frame_rgb.height()
            bpl = frame_rgb.bytesPerLine()

            ptr = frame_rgb.constBits()
            ptr.setsize(height * bpl)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, bpl))

            arr_valid = np.ascontiguousarray(arr[:, :width * 3])
            frame_bgr = cv2.cvtColor(arr_valid.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)

            processed_frame, found, corners = self.logic.process_live_frame(frame_bgr)
            self.current_corners = corners
            self.current_display_frame = processed_frame

            self.btn_capture.setEnabled(found)

            lbl_w, lbl_h = self.lbl_camera.width(), self.lbl_camera.height()
            if lbl_w > 10 and lbl_h > 10:
                scale = min(lbl_w / width, lbl_h / height)
                new_w, new_h = int(width * scale), int(height * scale)
                display_bgr = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_bgr = processed_frame

            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = display_rgb.shape
            final_qimg = QImage(display_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            self.lbl_camera.setPixmap(QPixmap.fromImage(final_qimg))

        except Exception as e:
            logging.error(f"Frame-Fehler im Kalibrierungs-Fenster: {e}")
        finally:
            self._unlock_worker()

    def _unlock_worker(self):
        try:
            if self.parent() and hasattr(self.parent(), 'worker'):
                self.parent().worker.lens_window_ready = True
        except:
            pass

    def _on_capture_clicked(self):
        count = self.logic.capture_calibration_frame(self.current_corners, self.current_display_frame)
        self.btn_capture.setText(f"📸 Bild aufnehmen ({count}/15)")

        if count >= 15:
            self.btn_capture.setStyleSheet("background-color: green; color: white;")
            self.lbl_info.setText("Genug Bilder gesammelt! Du kannst nun berechnen.")

    @pyqtSlot()
    def _on_calc_clicked(self):
        if self.logic.captured_images_count < 10:
            qtw.QMessageBox.warning(self, "Fehler", "Bitte sammeln Sie mindestens 10 Bilder.")
            return

        self.btn_calc.setEnabled(False)
        self.lbl_info.setText("Berechne Matrix... Bitte warten...")

        ret, mtx, dist, errors, rms_error = self.logic.calculate_camera_matrix()

        if ret:
            dlg = CalibrationResultDialog(self.logic.captured_images, mtx, dist, errors, self)
            if dlg.exec() == qtw.QDialog.DialogCode.Accepted:
                cam_type, ok = qtw.QInputDialog.getText(
                    self, "Globales Profil speichern",
                    "Wie soll dieses Linsen-Profil heißen? (z.B. 'Aukey_HD_Wide' oder 'Logitech_C920')"
                )
                print(ok,cam_type)
                if ok and cam_type:
                    profile_id = cam_type.lower().replace(" ", "_")
                    print("Der dis coeef speicehrung vorgne" ,dist.tolist())
                    ConfigManager.save_lens_profile(
                        camera_name=self.client_name,
                        profile_id=profile_id,
                        name=cam_type,
                        mtx=mtx.tolist(),
                        dist=dist.tolist(),
                        reprojection_error=rms_error
                    )

                    qtw.QMessageBox.information(
                        self, "Erfolg",
                        f"Profil '{cam_type}' wurde global gespeichert und für diese Kamera aktiviert."
                    )
                    self.accept()
                else:
                    self.btn_calc.setEnabled(True)
                    self.lbl_info.setText("Speichern abgebrochen.")
            else:
                self.btn_calc.setEnabled(True)
                self.lbl_info.setText("Ergebnis abgelehnt. Mehr Bilder sammeln?")
        else:
            qtw.QMessageBox.warning(self, "Fehler", "Kalibrierung fehlgeschlagen.")
            self.btn_calc.setEnabled(True)

    def accept(self):
        super().accept()

    def reject(self):
        super().reject()

    def closeEvent(self, event):
        ConfigManager.set_edit_mode(False)
        super().closeEvent(event)