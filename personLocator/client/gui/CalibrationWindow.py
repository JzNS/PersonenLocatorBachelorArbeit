import logging
import time
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QLabel,
    QVBoxLayout, QPushButton, QSizePolicy, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem, QGroupBox,
    QMessageBox, QCheckBox, QDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QEvent

from client.gui.logic.CalibrationWorker import CalibrationWorker
from client.gui.logic.SettingsDialog import SettingsDialog
from client.gui.logic.label.view.Calibration3DView import Calibration3DView
from client.utils.ConfigManager import ConfigManager
from client.gui.logic.label.utilityClasses.utilityClassSideTabs.CalibrationCoordinatesTable import CalibrationCoordinatesTable
from client.gui.logic.label.utilityClasses.utilityClassSideTabs.PersonSidebarWidget import PersonSidebarWidget

from client.gui.logic.label.LensCalibration import LensCalibrationWindow
from client.gui.logic.label.PrecisionWindowCalibration import PrecisionWindowCalibration
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client.network.logic.ClientCommandSender import CommandSender


class CalibrationWindow(QMainWindow):
    calibration_saved = pyqtSignal()

    def __init__(self, client_name: str):
        super().__init__()
        self.client_name = client_name
        self._is_syncing = False
        self.last_frame_size = (1280, 720)
        self.last_table_update = 0

        self.temp_points = []
        self.draw_mode = None

        self.selected_person_id = None

        self.sender: Optional['CommandSender'] = None
        self._setup_ui()
        self._setup_worker()

        self.setWindowTitle(f"PersonenLocator")
        self.resize(1300, 700)
        self.sync_ui_from_cache()
        self._on_view_option_changed()

    def set_network_sender(self, sender: 'CommandSender'):
        """Ermöglicht dem Controller, den Sender zu injizieren."""
        self.sender = sender

    def _send_tracking_data(self, person_results):
        """
        Sendet NUR noch die reinen 2D-Daten des YOLO-Modells an den Server.
        Jegliche 3D-Metriken und Winkel wurden entfernt.
        """
        clean_persons = []

        for p in person_results:
            clean_p = {
                "id": int(p.get('id', 0)),
                "status": str(p.get('status', 'Unbekannt')),
                "bbox": [int(x) for x in p.get('bbox', [0, 0, 0, 0])],
                "bbox_confidence": round(float(p.get('bbox_confidence', 0)), 2),
                "keypoints": p.get('keypoints', []),
                "metrics": p.get('metrics', {})
            }
            clean_persons.append(clean_p)

        if clean_persons and self.sender:
            self.sender.send_camera_update(self.client_name, clean_persons)
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)

        top_layout = QHBoxLayout()

        # 1. Sidebar
        self.sidebar = PersonSidebarWidget()
        self.sidebar.person_clicked.connect(self._on_person_selected)

        # 2. Kamera (Event Filter für Klicks)
        self.lbl_camera = self._create_display_label("Kamera")
        self.lbl_camera.setMouseTracking(True)
        self.lbl_camera.installEventFilter(self)

        # 3. 3D View
        self.view_3d = Calibration3DView("3D Modell")

        # 4. Rechtes Panel
        right_panel = self._create_right_panel()

        top_layout.addWidget(self.sidebar)
        top_layout.addWidget(self.lbl_camera, stretch=2)
        top_layout.addWidget(self.view_3d, stretch=2)
        top_layout.addWidget(right_panel, stretch=1)
        self.main_layout.addLayout(top_layout)

        self._setup_bottom_controls()

    def _create_right_panel(self) -> QWidget:
        """Erstellt das rechte Panel mit den Ansicht-Optionen und den Tabs für Skelett/Messwerte."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("3D Ansicht")
        vbox = QVBoxLayout()
        cam_config = ConfigManager.get_camera_settings(self.client_name)

        self.cb_render_active = QCheckBox("Rendering Aktiv")
        self.cb_render_active.setChecked(cam_config.get("view_render_enabled", True))
        self.cb_render_active.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.cb_render_active.toggled.connect(self._on_view_option_changed)
        vbox.addWidget(self.cb_render_active)

        # --- Performance Modus ---
        self.cb_perf_mode = QCheckBox("🚀 Performance Modus (Nur 2D)")
        self.cb_perf_mode.setChecked(cam_config.get("performance_mode", False))
        self.cb_perf_mode.setStyleSheet("font-weight: bold; color: #ffcc00;")
        self.cb_perf_mode.setToolTip("Deaktiviert alle 3D-Berechnungen (Raycasting, Metriken) für max. FPS.")
        self.cb_perf_mode.toggled.connect(self._on_view_option_changed)
        vbox.addWidget(self.cb_perf_mode)

        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #555;")
        vbox.addWidget(line)
        self.cb_show_real = QCheckBox("Raum")
        self.cb_show_real.setChecked(cam_config.get("view_show_real", True))
        self.cb_show_real.toggled.connect(self._on_view_option_changed)

        self.cb_show_cam = QCheckBox("Gitter")
        self.cb_show_cam.setChecked(cam_config.get("view_show_grid", True))
        self.cb_show_cam.toggled.connect(self._on_view_option_changed)

        self.cb_show_skel = QCheckBox("3D Skelett")
        self.cb_show_skel.setChecked(cam_config.get("view_show_skeleton", True))
        self.cb_show_skel.toggled.connect(self._on_view_option_changed)

        self.cb_show_sightlines = QCheckBox("Sichtlinien (Kamera)")
        self.cb_show_sightlines.setChecked(cam_config.get("view_show_sightlines", True))
        self.cb_show_sightlines.toggled.connect(self._on_view_option_changed)

        self.cb_show_grid = QCheckBox("Boden-Gitter")
        self.cb_show_grid.setChecked(cam_config.get("view_show_floor_grid", True))
        self.cb_show_grid.toggled.connect(self._on_view_option_changed)

        self.cb_show_rays = QCheckBox("Präzisions-Strahlen")
        self.cb_show_rays.setChecked(cam_config.get("view_show_rays", True))
        self.cb_show_rays.toggled.connect(self._on_view_option_changed)


        vbox.addWidget(self.cb_show_real)
        vbox.addWidget(self.cb_show_cam)
        vbox.addWidget(self.cb_show_grid)
        vbox.addWidget(self.cb_show_rays)
        vbox.addWidget(self.cb_show_skel)
        vbox.addWidget(self.cb_show_sightlines)

        grp.setLayout(vbox)
        layout.addWidget(grp)
        layout.addWidget(grp)
        #Tabs
        self.tabs_right = QTabWidget()

        self.table_skeleton = CalibrationCoordinatesTable()
        self.table_skeleton.value_changed.connect(self._on_table_value_changed)

        self.tabs_measurements = QTabWidget()
        self.corner_tables = {}
        for label in ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]:
            t = QTableWidget(0, 3)
            t.verticalHeader().setVisible(False)
            t.setHorizontalHeaderLabels(["Typ", "Name", "Wert"])
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.corner_tables[label] = t
            self.tabs_measurements.addTab(t, label)

        self.tabs_right.addTab(self.table_skeleton, "Skelett")
        self.tabs_right.addTab(self.tabs_measurements, "Messwerte")

        layout.addWidget(self.tabs_right)
        return container

    def _setup_worker(self):
        """Initialisiert den CalibrationWorker, verbindet die Signale und startet den Thread."""
        self.worker = CalibrationWorker(self.client_name)
        self.worker.image_data.connect(self.update_frames)
        self.worker.measurement_data.connect(self.update_measurements_table)

        self.view_3d.rotation_requested.connect(self.worker.update_rotation)
        self.view_3d.zoom_requested.connect(self.worker.update_zoom)

        if hasattr(self.worker, 'ai_thread'):
            self.worker.ai_thread.learn_data_ready.connect(self.on_learn_data_ready)
        else:
            logging.error("CalibrationWorker hat kein 'ai_thread' Attribut! Sync geht nicht.")
        self.worker.toolbox.zones_changed.connect(self._sync_zones_to_server)
        self.worker.start()

    @pyqtSlot()
    def _sync_zones_to_server(self):
        """Wird aufgerufen, wenn in der Toolbox Zonen gezeichnet oder gelöscht werden."""
        if self.sender:
            full_settings = ConfigManager.get_camera_settings(self.client_name)
            self.sender.send_db_camera_settings(self.client_name, full_settings)
            self.lbl_calib_instr.setText("Zonen-Update an Server gesendet.")

    def _on_start_precision_calib(self):
        """Öffnet das Präzisions-Fenster, sperrt aber bei Performance-Modus."""
        if self.cb_perf_mode.isChecked():
            QMessageBox.warning(self, "Gesperrt",
                "Bitte schalte zuerst den Performance-Modus aus (Häkchen entfernen), "
                "damit die Vierecke vom Server geladen werden können!")
            return

        self.precision_window = PrecisionWindowCalibration(self.client_name, self)
        self.worker.set_precision_mode(True)
        self.worker.full_res_image_data.connect(self.precision_window.update_frame)
        self.precision_window.finished.connect(self._on_precision_calib_closed)
        self.precision_window.show()
    def _on_precision_calib_closed(self):
        """Wird aufgerufen, wenn das Precision-Fenster geschlossen wird."""
        self.worker.set_precision_mode(False)

        self.worker.lens_window_ready = True

        try:
            self.worker.full_res_image_data.disconnect(self.precision_window.update_frame)
        except:
            pass

        self.precision_window = None
        self.worker.trigger_config_reload()
    def on_learn_data_ready(self, p_id: int, stats: dict):
        """Wird aufgerufen, wenn der AI-Thread neue Lern-Daten hat. Sendet diese Daten an den Server."""
        if self.sender:
            self.sender.send_learned_body_data(p_id, stats)

    def _on_person_selected(self, p_id):
        """Reagiert auf Klicks in der Personenliste (Sidebar) und setzt die ausgewählte Person."""
        self.selected_person_id = p_id
        self.sidebar.set_selected_id(p_id)

    @pyqtSlot(object)
    def update_measurements_table(self, data: dict):
        """Aktualisiert die rechte Seite mit den Messwerten und der Personenliste (Vom Worker aufgerufen)."""
        persons = data.get("persons", [])
        if self.sender:
            self._send_tracking_data(persons)
        self.sidebar.update_persons(persons)

        target_person = None
        if persons:
            if self.selected_person_id is not None:
                for p in persons:
                    if p['id'] == self.selected_person_id:
                        target_person = p
                        break

            if target_person is None:
                target_person = persons[0]
                self.selected_person_id = target_person['id']
                self.sidebar.set_selected_id(self.selected_person_id)

        if hasattr(self, 'table_skeleton'):
            if target_person:
                self.table_skeleton.setHorizontalHeaderLabels([f"Person {target_person['id']}", "Wert", "Status"])
                self.table_skeleton.update_skeleton_data(
                    target_person.get("keypoints", []),
                    target_person.get("metrics", {})
                )
            else:
                self.table_skeleton.setRowCount(0)

        now = time.time()
        if now - self.last_table_update > 0.03:
            self.last_table_update = now
            if self.tabs_right.currentIndex() == 1:
                self._update_geometry_tabs(data.get("geometry", {}))



    @pyqtSlot(QImage, QImage)
    def update_frames(self, main_img: QImage, p3d_img: QImage):
        """Aktualisiert die Bildschirme (Vom Worker aufgerufen)."""
        if getattr(self, '_is_processing_frames', False):
            return

        self._is_processing_frames = True

        try:
            if not main_img.isNull():
                self._set_pixmap_scaled(self.lbl_camera, main_img)

            if not p3d_img.isNull():
                self.view_3d.update_frame(p3d_img)

        except Exception as e:
            logging.error(f"Fehler im GUI-Update: {e}")
        finally:
            self._is_processing_frames = False

    def _map_click_to_frame(self, pos):
            """
            Maps a click from the QLabel coordinates to the FULL RESOLUTION camera frame.
            """
            cam_settings = ConfigManager.get_camera_settings(self.client_name)
            real_res = cam_settings.get("resolution", [1280, 720])
            fw, fh = real_res[0], real_res[1]

            lw, lh = self.lbl_camera.width(), self.lbl_camera.height()

            # Nicht durch 0 teilen
            if fw == 0 or fh == 0 or lw == 0 or lh == 0: return None

            scale = min(lw / fw, lh / fh)

            ox = (lw - fw * scale) / 2
            oy = (lh - fh * scale) / 2

            rx = int((pos.x() - ox) / scale)
            ry = int((pos.y() - oy) / scale)

            if 0 <= rx < fw and 0 <= ry < fh:
                return (rx, ry)
            return None

    def _set_pixmap_scaled(self, label: QLabel, img):
        """
        Optimierte Anzeige: Skaliert QImage auf CPU, um GUI-Blockaden zu vermeiden.
        """
        if img is None: return

        if label.width() < 10 or label.height() < 10:
            return

        target_size = label.size()

        if isinstance(img, QImage):
            if img.width() != target_size.width() and not img.isNull():
                img = img.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )

            label.setPixmap(QPixmap.fromImage(img))

        elif isinstance(img, QPixmap):
            pm = img.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            label.setPixmap(pm)

    def _on_table_value_changed(self, row, axis, value):
        """Reagiert nur noch auf die manuelle Größen-Korrektur aus der Skelett-Tabelle."""
        if not hasattr(self, 'worker'): return

        if axis == "fixed_height":
            if self.selected_person_id is not None:
                self.worker.set_person_height(self.selected_person_id, value)
            elif self.worker.last_person_results:
                p_id = self.worker.last_person_results[0]['id']
                self.worker.set_person_height(p_id, value)

    def _on_view_option_changed(self):
        """Reagiert auf Änderungen der Ansicht-Optionen (z.B. Performance-Modus) und synchronisiert diese Änderungen sofort mit dem Server."""
        if getattr(self, '_is_syncing', False):
            return
        if hasattr(self, 'worker') and hasattr(self.worker, 'toolbox'):
            opts = self.worker.toolbox.view_options

            was_perf_mode = opts.get("performance_mode", False)
            is_perf_mode = self.cb_perf_mode.isChecked()

            # 1. Sofortige UI-Optionen übernehmen
            opts["render_3d_enabled"] = self.cb_render_active.isChecked()
            opts["performance_mode"] = is_perf_mode
            opts["show_real_world"] = self.cb_show_real.isChecked()
            opts["show_floor_grid"] = self.cb_show_grid.isChecked()
            opts["show_rays"] = self.cb_show_rays.isChecked()
            opts["show_camera_world"] = self.cb_show_cam.isChecked()
            opts["show_skeleton_3d"] = self.cb_show_skel.isChecked()
            opts["show_sightlines"] = self.cb_show_sightlines.isChecked()

            view_settings = {
                "performance_mode": is_perf_mode,
                "view_render_enabled": self.cb_render_active.isChecked(),
                "view_show_real": self.cb_show_real.isChecked(),
                "view_show_grid": self.cb_show_cam.isChecked(),
                "view_show_floor_grid": self.cb_show_grid.isChecked(),
                "view_show_rays": self.cb_show_rays.isChecked(),
                "view_show_skeleton": self.cb_show_skel.isChecked(),
                "view_show_sightlines": self.cb_show_sightlines.isChecked()
            }
            ConfigManager.update_camera_settings(self.client_name, view_settings)
            if was_perf_mode and not is_perf_mode:
                self.sender.send_message("SERVER", "REQUEST_CONFIG_FULL")

            if was_perf_mode and not is_perf_mode:
                if hasattr(self, 'sender') and self.sender:
                    self.sender.send_register()
                    self.lbl_calib_instr.setText("Lade 3D-Daten live vom Server...")

            color = "#00ff00" if self.cb_render_active.isChecked() else "#ff0000"
            self.cb_render_active.setStyleSheet(f"font-weight: bold; color: {color};")



    def _update_geometry_tabs(self, geometry: dict):
        """Aktualisiert die Tabellen in den Geometrie-Tabs basierend auf den neuesten Daten vom Worker."""
        label = self.tabs_measurements.tabText(self.tabs_measurements.currentIndex())
        entries = geometry.get(label, [])
        table = self.corner_tables.get(label)
        if table:
            table.setRowCount(len(entries))
            for r, item in enumerate(entries):
                table.setItem(r, 0, QTableWidgetItem(str(item["type"])))
                table.setItem(r, 1, QTableWidgetItem(str(item["name"])))
                table.setItem(r, 2, QTableWidgetItem(str(item["value"])))

    def save_calibration(self):
        try:
            # Checkboxen auslesen
            view_settings = {
                "view_render_enabled": self.cb_render_active.isChecked(),
                "performance_mode": self.cb_perf_mode.isChecked(),
                "view_show_real": self.cb_show_real.isChecked(),
                "view_show_grid": self.cb_show_cam.isChecked(),
                "view_show_floor_grid": self.cb_show_grid.isChecked(),
                "view_show_rays": self.cb_show_rays.isChecked(),
                "view_show_skeleton": self.cb_show_skel.isChecked(),
                "view_show_sightlines": self.cb_show_sightlines.isChecked()
            }

            ConfigManager.update_camera_settings(self.client_name, view_settings)

            if self.sender:
                full_settings = ConfigManager.get_camera_settings(self.client_name)
                self.sender.send_db_camera_settings(self.client_name, full_settings)

            QMessageBox.information(self, "Erfolg", "Kalibrierung & Ansicht gespeichert und an Server gesendet.")
            self.calibration_saved.emit()

        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))


    def showEvent(self, event):
        """Wird automatisch von PyQt aufgerufen, sobald das Fenster auf dem Bildschirm erscheint."""
        super().showEvent(event)
        self.sync_ui_from_cache()

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait()
        e.accept()

    def _create_button(self, txt, cb, is_danger=False):
        b = QPushButton(txt);
        b.setFixedHeight(40);
        b.clicked.connect(cb)
        if is_danger: b.setStyleSheet("background-color: #8B0000; color: white;")
        return b

    def _create_display_label(self, txt):
        """Erstellt ein QLabel mit Standard-Styling für die Kamera- und 3D-Ansicht."""
        l = QLabel(txt);
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.setStyleSheet("background: black; color: white; border: 2px solid #333;")
        l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        return l


    def eventFilter(self, src, evt):
        """Fängt Klicks auf das Kamerabild ab, um Zonen zu zeichnen oder zu löschen."""
        if src == self.lbl_camera and evt.type() == QEvent.Type.MouseButtonPress:

            # Zeichnen von Dead/Mirror Zonen
            if self.draw_mode:
                lw, lh = self.lbl_camera.width(), self.lbl_camera.height()
                real_res = ConfigManager.get_camera_settings(self.client_name).get("resolution", [1280, 720])
                fw, fh = real_res[0], real_res[1]

                if lw > 0 and lh > 0:
                    scale = min(lw / fw, lh / fh)
                    ox = (lw - fw * scale) / 2
                    oy = (lh - fh * scale) / 2

                    rx = int((evt.pos().x() - ox) / scale)
                    ry = int((evt.pos().y() - oy) / scale)

                    if 0 <= rx < fw and 0 <= ry < fh:
                        if evt.button() == Qt.MouseButton.LeftButton:
                            self.temp_points.append((rx, ry))
                            count = len(self.temp_points)

                            if count == 4:
                                self.worker.add_zone(self.draw_mode, self.temp_points)
                                self.lbl_calib_instr.setText(f"Zone ({self.draw_mode}) gespeichert!")
                                self.draw_mode = None
                                self.temp_points = []
                                self.lbl_camera.setCursor(Qt.CursorShape.ArrowCursor)
                            else:
                                self.lbl_calib_instr.setText(f"Punkt {count}/4 gesetzt...")

                        elif evt.button() == Qt.MouseButton.RightButton:
                            self.draw_mode = None
                            self.temp_points = []
                            self.lbl_camera.setCursor(Qt.CursorShape.ArrowCursor)
                            self.lbl_calib_instr.setText("Zeichnen abgebrochen.")

                        return True

        return super().eventFilter(src, evt)


    def _on_open_settings(self):
        """Öffnet den Settings-Dialog, um Kamera- und Raum-Settings zu bearbeiten."""
        current_settings = ConfigManager.get_camera_settings(self.client_name)
        dialog = SettingsDialog(self, self.client_name, current_settings)
        dialog.capacity_changed.connect(self._update_capacity_live)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            converted_data = dialog.get_data()

            try:
                ConfigManager.update_camera_settings(self.client_name, converted_data)

                if self.sender:
                    full_settings = ConfigManager.get_camera_settings(self.client_name)
                    self.sender.send_db_camera_settings(self.client_name, full_settings)

                    if "room_dimensions" in converted_data:
                        dims = converted_data["room_dimensions"]
                        self.sender.send_db_global_room(dims["width"], dims["height"], dims["depth"])

                self.worker.trigger_config_reload()

                self.lbl_calib_instr.setText(f"Setup gespeichert (Profil: {converted_data['active_lens_profile']})")
            except Exception as e:
                logging.error(f"Fehler beim Übernehmen der Settings: {e}")
                QMessageBox.warning(self, "Fehler", f"Daten konnten nicht übernommen werden: {e}")

    def _update_capacity_live(self, val):
        """Aktualisiert den Wert in der Toolbox sofort für den laufenden AsyncDetector."""
        self.worker.ai_thread.reset_tracking()
        self.worker.toolbox.view_options["render_capacity"] = val  #
        self.lbl_calib_instr.setText(f"KI-Render-Last: {val}%")

    def _on_start_camera_calib(self):
        """Öffnet das neue Fenster für die Schachbrett-Linsenkalibrierung."""
        self.lens_window = LensCalibrationWindow(self.client_name, self)
        self.worker.set_precision_mode(True)
        self.worker.full_res_image_data.connect(self.lens_window.update_frame)
        self.lens_window.finished.connect(self._on_camera_calib_closed)
        self.lens_window.show()

    @pyqtSlot()
    def sync_ui_from_cache(self):
        """Aktualisiert die UI-Elemente, ohne einen Speicherbefehl auszulösen."""
        self._is_syncing = True #Sperren

        cam_config = ConfigManager.get_camera_settings(self.client_name)
        self.cb_render_active.setChecked(cam_config.get("view_render_enabled", True))
        self.cb_perf_mode.setChecked(cam_config.get("performance_mode", False))
        self.cb_show_real.setChecked(cam_config.get("view_show_real", True))
        self.cb_show_cam.setChecked(cam_config.get("view_show_grid", True))
        self.cb_show_skel.setChecked(cam_config.get("view_show_skeleton", True))
        self.cb_show_sightlines.setChecked(cam_config.get("view_show_sightlines", True))
        self.cb_show_grid.setChecked(cam_config.get("view_show_floor_grid", True))
        self.cb_show_rays.setChecked(cam_config.get("view_show_rays", True))

        self._on_view_option_changed()

        self._is_syncing = False #Sperre aufheben
        if hasattr(self, 'worker') and self.worker:
            self.worker.trigger_config_reload()

        logging.info("UI-Checkboxen synchronisiert & Worker hat Zonen/PnP geladen.")


    def _on_camera_calib_closed(self):
        """Wird automatisch aufgerufen, wenn die Linsenkalibrierung beendet wird."""
        self.worker.set_precision_mode(False)
        self.worker.full_res_image_data.disconnect(self.lens_window.update_frame)
        self.lens_window = None
        self.worker.trigger_config_reload()
        self.lbl_calib_instr.setText("Linsenkalibrierung beendet. Config neu geladen.")
    def _setup_bottom_controls(self):
        """Erstellt die untere Reihe mit den Buttons für die
        Kalibrierung und das Speichern. Die Zonen-Buttons wurden entfernt,
        da sie jetzt über die Toolbar der Kamera-Ansicht zugänglich sind."""
        wiz_layout = QHBoxLayout()
        self.btn_start_precision = self._create_button("🎯 Start Precision Kalibrierung", self._on_start_precision_calib)
        self.btn_start_camera = self._create_button("🎯 Start Kamera Kalibrierung", self._on_start_camera_calib)
        self.lbl_calib_instr = QLabel("Bereit.")
        self.lbl_calib_instr.setStyleSheet("color: #00ff7f; font-weight: bold; margin-left: 10px;")
        wiz_layout.addWidget(self.btn_start_camera)
        wiz_layout.addWidget(self.btn_start_precision)
        wiz_layout.addWidget(self.lbl_calib_instr)
        wiz_layout.addStretch()
        self.main_layout.addLayout(wiz_layout)
        sys_layout = QHBoxLayout()
        self.btn_save = self._create_button("Alles Speichern", self.save_calibration)

        self.btn_settings = self._create_button("⚙️ Settings", self._on_open_settings)
        self.btn_settings.setFixedWidth(120)

        sys_layout.addWidget(self.btn_save)
        sys_layout.addStretch()

        sys_layout.addWidget(self.btn_settings)
        self.main_layout.addLayout(sys_layout)