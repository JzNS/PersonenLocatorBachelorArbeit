import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSlot

from client.gui.logic.label.utilityClasses.utilityClassLensCalibration.CalibrationResultDialog import \
    CalibrationResultDialog
from client.gui.logic.label.utilityClasses.utilityClassLensCalibration.LensCalibrationLogic import \
    LensCalibrationLogic
# Importiere deine Logik-Klasse (Pfad ggf. anpassen)
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

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 1. Kamera Label
        self.lbl_camera = QLabel("Warte auf Kamera...")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background: black;")
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl_camera, stretch=1)

        # 2. Info Label
        self.lbl_info = QLabel("Halte das Schachbrett in die Kamera. Mindestens 15 Bilder benötigt!")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: yellow;")
        layout.addWidget(self.lbl_info)

        # 3. Buttons
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
        """Wird vom Worker gefeuert. Nur wenn dieses Bild fertig ist, darf das nächste kommen!"""
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

            # 2. Suche Schachbrett
            processed_frame, found, corners = self.logic.process_live_frame(frame_bgr)
            self.current_corners = corners
            self.current_display_frame = processed_frame

            self.btn_capture.setEnabled(found)

            # 3. BILD FÜR DIE GUI VERKLEINERN
            lbl_w, lbl_h = self.lbl_camera.width(), self.lbl_camera.height()
            if lbl_w > 10 and lbl_h > 10:
                scale = min(lbl_w / width, lbl_h / height)
                new_w, new_h = int(width * scale), int(height * scale)
                display_bgr = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_bgr = processed_frame

            # 4. BILD ANZEIGEN
            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = display_rgb.shape
            final_qimg = QImage(display_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            self.lbl_camera.setPixmap(QPixmap.fromImage(final_qimg))

        except Exception as e:
            import logging
            logging.error(f"Frame-Fehler im Kalibrierungs-Fenster: {e}")
        finally:
            self._unlock_worker()

    def _unlock_worker(self):
        """Gibt das Signal an den Worker, dass die GUI wieder Zeit für den nächsten Frame hat."""
        try:
            if self.parent() and hasattr(self.parent(), 'worker'):
                self.parent().worker.lens_window_ready = True
        except:
            pass

    def _on_capture_clicked(self):
        """Nimmt das aktuelle Brett in die Matrix-Berechnung auf."""
        count = self.logic.capture_calibration_frame(self.current_corners, self.current_display_frame)
        self.btn_capture.setText(f"📸 Bild aufnehmen ({count}/15)")

        if count >= 15:
            self.btn_capture.setStyleSheet("background-color: green; color: white;")
            self.lbl_info.setText("Genug Bilder gesammelt! Du kannst nun berechnen.")



    def _on_calc_clicked(self):
        self.lbl_info.setText("Berechne Matrix... Bitte warten!")
        self.btn_calc.setEnabled(False)

        import PyQt6.QtWidgets as qtw
        qtw.QApplication.processEvents()

        success, mtx, dist, errors = self.logic.calculate_camera_matrix()

        if success:
            dialog = CalibrationResultDialog(self.logic.captured_images, mtx, dist, errors, self)

            if dialog.exec() == qtw.QDialog.DialogCode.Accepted:

                cam_type, ok = qtw.QInputDialog.getText(
                    self,
                    "Kamera-Modell speichern",
                    "Gib den Typ/Namen dieser Kamera ein\n(z.B. 'Logitech_C920' oder 'GoPro_Hero10'):",
                    qtw.QLineEdit.EchoMode.Normal,
                    "Standard_Webcam"
                )

                if ok and cam_type.strip():
                    cam_type = cam_type.strip()


                    cam_settings = ConfigManager.get_camera_settings(self.client_name)

                    new_lens_data = {
                        "active_lens_profile": cam_type,
                        "camera_matrix": mtx.tolist(),
                        "dist_coeffs": dist.tolist()
                    }

                    ConfigManager.update_camera_settings(self.client_name, new_lens_data)

                    if self.parent() and hasattr(self.parent(), 'sender') and self.parent().sender:
                        full_settings = ConfigManager.get_camera_settings(self.client_name)
                        self.parent().sender.send_db_camera_settings(self.client_name, full_settings)

                    qtw.QMessageBox.information(
                        self,
                        "Erfolg",
                        f"Linsen-Profil '{cam_type}' wurde für {self.client_name} gespeichert und an die DB gesendet!"
                    )

                    if self.parent() and hasattr(self.parent(), 'worker'):
                        self.parent().worker.trigger_config_reload()

                    self.accept()
                else:
                    self.btn_calc.setEnabled(True)
                    self.lbl_info.setText("Speichern abgebrochen (Kein Name eingegeben).")
            else:
                self.btn_calc.setEnabled(True)
                self.lbl_info.setText("Speichern abgebrochen. Du kannst weiter aufnehmen.")
        else:
            qtw.QMessageBox.warning(self, "Fehler", "Nicht genug Bilder gesammelt.")
            self.btn_calc.setEnabled(True)