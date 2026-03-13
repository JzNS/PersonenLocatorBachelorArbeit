import sys
import datetime
import numpy as np
import cv2
from typing import Dict
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QTextEdit, QGroupBox, QHeaderView, QGridLayout, QHBoxLayout, QPushButton, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt
from server.gui.logic.PersonDetailsWindow import PersonDetailsWindow
from server.gui.logic.FusionWizard import FusionWizard
from server.gui.logic.Server3DView import Server3DView
from server.gui.logic.data.utilityClassDatabase.DatabaseViewerWindow import DatabaseViewerWindow
from server.utils.ConfigManager import ConfigManager
from server.gui.logic.GeometryMath import GeometryMath
import time


class ServerDashboard(QMainWindow):
    """
    Die grafische Benutzeroberfläche des Servers.
    Layout: Links (Kameras) | Rechts (Infos)
    """
    # Signale
    sig_update_camera_view = pyqtSignal(str, list)
    sig_register_client = pyqtSignal(str, str)
    sig_update_heartbeat = pyqtSignal(str)
    sig_set_offline = pyqtSignal(str)
    sig_log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Server Dashboard - Master Control")
        self.resize(1600, 900)

        self.controller = None
        self.fusion_wizard_class = FusionWizard
        self.details_window = None

        self.camera_views = {}
        self.view_master = None
        self.__client_items: Dict[str, QTreeWidgetItem] = {}

        # Cache für ALLE Personen
        self.person_cache = {
            "CAMERA_1": [], "CAMERA_2": [], "CAMERA_3": [], "CAMERA_4": []
        }

        # 1. Config laden
        self.master_config = ConfigManager.get_master_config_data()

        # 2. UI aufbauen
        self.__setup_ui()
        self.__connect_signals()

        # 3. Initialisieren
        self.__initialize_static_views()

    def set_controller(self, controller_instance):
        self.controller = controller_instance

    def show_window(self) -> None:
        self.show()

    def __setup_ui(self) -> None:
        """Erstellt die gesamte GUI-Struktur mit Platzhaltern. Alle dynamischen Daten werden später über Signale aktualisiert."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ==================================================
        # 1. Master Grid
        # ==================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        master_group = QGroupBox("MASTER VIEW (Gesamt)")
        master_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; color: #444; }")
        master_layout = QVBoxLayout(master_group)
        master_layout.setContentsMargins(2, 10, 2, 2)

        self.view_master = Server3DView("GESAMTANSICHT")
        self.view_master.setStyleSheet("background-color: #050505; border: 2px solid #555;")
        master_layout.addWidget(self.view_master)

        left_layout.addWidget(master_group, stretch=5)

        grid_group = QGroupBox("Einzelkameras")
        grid_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        grid_layout = QGridLayout(grid_group)
        grid_layout.setContentsMargins(2, 10, 2, 2)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i in range(1, 5):
            cam_name = f"CAMERA_{i}"
            view = Server3DView(cam_name)
            self.camera_views[cam_name] = view
            row, col = positions[i - 1]
            grid_layout.addWidget(view, row, col)

        left_layout.addWidget(grid_group, stretch=4)
        main_layout.addWidget(left_widget, stretch=4)

        # ==================================================
        # 2. Clients + Log
        # ==================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        group_clients = QGroupBox("Verbundene Clients")
        layout_clients = QVBoxLayout(group_clients)

        self.tree_clients = QTreeWidget()
        self.tree_clients.setHeaderLabels(["Client", "IP", "Status", "Ping"])

        self.tree_clients.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_clients.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree_clients.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_clients.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.tree_clients.setStyleSheet("font-size: 12px;")
        layout_clients.addWidget(self.tree_clients)

        right_layout.addWidget(group_clients, stretch=1)

        # --- Buttons ---
        btn_wizard = QPushButton("🛠 Fusion Wizard (Person zuweisen)")
        btn_wizard.setStyleSheet("background-color: #0055AA; color: white; padding: 10px; font-weight: bold;")
        btn_wizard.clicked.connect(self.open_fusion_wizard)

        btn_details = QPushButton("📊 Live Personen-Metriken")
        btn_details.setStyleSheet("background-color: #AA5500; color: white; padding: 10px; font-weight: bold;")
        btn_details.clicked.connect(self.open_details_window)

        btn_db_viewer = QPushButton("🗄️ System-Datenbank ansehen")
        btn_db_viewer.setStyleSheet("background-color: #444444; color: white; padding: 10px; font-weight: bold;")
        btn_db_viewer.clicked.connect(self.open_database_viewer)

        btn_sql_log = QPushButton("🔍 SQL-Inspektor umschalten")
        btn_sql_log.setStyleSheet("background-color: #2b7a0b; color: white; padding: 10px; font-weight: bold;")
        btn_sql_log.clicked.connect(self.toggle_sql_logging)

        btn_cam_pos = QPushButton("📷 Kamera 3D-Positionen berechnen")
        btn_cam_pos.setStyleSheet("background-color: #6a0dad; color: white; padding: 10px; font-weight: bold;")
        btn_cam_pos.clicked.connect(self.show_camera_positions)

        right_layout.addWidget(btn_wizard)
        right_layout.addWidget(btn_details)
        right_layout.addWidget(btn_db_viewer)
        right_layout.addWidget(btn_sql_log)
        right_layout.addWidget(btn_cam_pos)

        group_log = QGroupBox("System Events")
        layout_log = QVBoxLayout(group_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #f8f8f8; font-family: Consolas; font-size: 11px; color: #333;")
        layout_log.addWidget(self.txt_log)

        right_layout.addWidget(group_log, stretch=1)

        right_widget.setMinimumWidth(300)
        right_widget.setMaximumWidth(450)

        main_layout.addWidget(right_widget, stretch=1)

    def show_camera_positions(self):
        """Berechnet live die 3D Position der Kamera und zeigt sie an."""
        if not self.controller or not self.controller.config_cache:
            QMessageBox.information(self, "Kamera Positionen", "Noch keine Kameras verbunden oder berechnet.")
            return

        info_text = "📍 BERECHNETE KAMERA-POSITIONEN (Im 3D-Raum)\n"
        info_text += "="*65 + "\n\n"

        found_any = False

        for cam_name, cam_conf in self.controller.config_cache.items():
            if not cam_name.startswith("CAMERA"):
                continue

            custom_rects = cam_conf.get("custom_rectangles", [])
            res = cam_conf.get("resolution", [1920, 1080])
            cam_matrix = cam_conf.get("camera_matrix")
            dist_c = cam_conf.get("dist_coeffs")

            pose = GeometryMath.get_camera_pose([], [], custom_rectangles=custom_rects,
                                                img_size=res, camera_matrix_override=cam_matrix,
                                                dist_coeffs_override=dist_c)
            if pose:
                found_any = True
                rvec, tvec, K, dist = pose
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = -np.dot(R.T, tvec).flatten()

                sy = np.sqrt(R.T[0, 0] ** 2 + R.T[1, 0] ** 2)
                pitch = np.arctan2(R.T[2, 1], R.T[2, 2]) if sy > 1e-6 else np.arctan2(-R.T[1, 2], R.T[1, 1])
                yaw = np.arctan2(-R.T[2, 0], sy)
                roll = np.arctan2(R.T[1, 0], R.T[0, 0]) if sy > 1e-6 else 0
                deg = np.degrees([pitch, yaw, roll])

                info_text += f"🎥 {cam_name}:\n"
                info_text += f"   Koordinaten : X: {cam_pos[0]:8.1f} cm | Y: {cam_pos[1]:8.1f} cm | Z: {cam_pos[2]:8.1f} cm\n"
                info_text += f"   Winkel      : Pitch (Neigung): {deg[0]:5.1f}° | Yaw (Schwenk): {deg[1]:5.1f}° | Roll: {deg[2]:5.1f}°\n"
                info_text += "-"*65 + "\n"

        if not found_any:
            info_text += "Für keine Kamera konnten gültige Vierecke berechnet werden.\nBitte zeichne zuerst Boden-Vierecke im Client!"

        msg = QMessageBox(self)
        msg.setWindowTitle("Kamera 3D Positionen")
        msg.setText(info_text)
        msg.setStyleSheet("QLabel{font-family: Consolas; font-size: 14px;}")
        msg.exec()

    def toggle_sql_logging(self):
        """Schaltet den SQL-Inspektor ein oder"""
        if not self.controller or not hasattr(self.controller, 'system_db'):
            self.log_message("Fehler: Datenbank-Controller nicht bereit.")
            return

        is_active = self.controller.system_db.toggle_sql_logging()
        status = "AKTIVIERT" if is_active else "DEAKTIVIERT"
        self.log_message(f"SQL-Inspektor wurde {status}.")

    def open_fusion_wizard(self):
        """Öffnet den Fusion Wizard, um Personen manuell zuzuordnen oder zu verschieben."""
        if not self.controller:
            self.log_message("Fehler: GUI hat keinen Zugriff auf Controller.")
            return
        dlg = FusionWizard(self.controller, self)
        dlg.exec()

    def open_database_viewer(self):
        """Öffnet ein neues Fenster, das die Datenbanktabellen anzeigt und einfache Abfragen ermöglicht."""
        if not self.controller:
            self.log_message("Fehler: Datenbank noch nicht bereit.")
            return
        if not hasattr(self.controller, 'db_viewer') or not self.controller.db_viewer:
            self.controller.db_viewer = DatabaseViewerWindow(self.controller, self)
        self.controller.db_viewer.show()
        self.controller.db_viewer.load_table_names()

    def open_details_window(self):
        """Öffnet ein neues Fenster, das detaillierte Metriken und Eigenschaften aller Personen anzeigt, die aktuell im System verfolgt werden."""
        if not self.details_window:
            self.details_window = PersonDetailsWindow(self)
        self.details_window.show()

    def __connect_signals(self) -> None:
        """Verbindet die internen Signale mit den entsprechenden Slots, um eine thread-sichere Aktualisierung der GUI zu gewährleisten."""
        self.sig_register_client.connect(self.__on_register_client)
        self.sig_update_heartbeat.connect(self.__on_update_heartbeat)
        self.sig_set_offline.connect(self.__on_set_offline)
        self.sig_log_message.connect(self.__on_log_message)
        self.sig_update_camera_view.connect(self.__on_update_camera_view)

    def __initialize_static_views(self):
        """Initialisiert die 3D-Ansichten mit leeren Szenen, damit sie sofort sichtbar sind, auch wenn noch keine Daten von den Clients kommen. Das verhindert leere oder fehlerhafte Darstellungen beim Start."""
        self.log_message("Initialisiere 3D-Ansichten...")
        for cam_name in self.camera_views.keys():
            self.__on_update_camera_view(cam_name, [])

    def register_client(self, name: str, ip: str) -> None:
        self.sig_register_client.emit(name, ip)

    def update_heartbeat(self, name: str) -> None:
        self.sig_update_heartbeat.emit(name)

    def set_client_offline(self, name: str) -> None:
        self.sig_set_offline.emit(name)

    def log_message(self, message: str) -> None:
        self.sig_log_message.emit(message)

    def update_camera_data(self, camera_name: str, person_list: list):
        self.sig_update_camera_view.emit(camera_name, person_list)

    def __on_update_camera_view(self, cam_name: str, persons: list):
        if not self.controller:
            return

        current_config = self.controller.config_cache
        def get_main_room_points(custom_rects):
            """Sucht in den custom_rectangles nach einem Eintrag mit "Haupt-Kalibrierung"
            oder "MAIN_ROOM_CALIB" und extrahiert die 8 Eckpunkte in der Reihenfolge
            HUL, HUR, HOL, HOR, VUL, VOL, VUR, VOR.
            Gibt eine Liste von 3D-Punkten zurück oder eine leere Liste, wenn nicht gefunden."""
            point_labels = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]
            for rect in custom_rects:
                if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("type") == "Haupt-Kalibrierung":
                    corners = rect.get("corners", [])
                    ordered_pts = []
                    for lbl in point_labels:
                        found = next((c for c in corners if c.get("label") == lbl), None)
                        if found:
                            ordered_pts.append(np.array([found["x"], found["y"], found["z"]], dtype=np.float32))
                    if len(ordered_pts) == 8: return ordered_pts
            return []

        # ==========================================
        # A. Einzelne Kamera Views
        # ==========================================
        if cam_name in self.camera_views:
            view = self.camera_views[cam_name]
            cam_conf = current_config.get(cam_name, {})

            room_dims = cam_conf.get("room_dimensions", {"width": 600, "height": 250, "depth": 800})
            cam_pos_label = cam_conf.get("position", None)
            custom_rects = cam_conf.get("custom_rectangles", [])
            res = cam_conf.get("resolution", [1920, 1080])
            cam_matrix = cam_conf.get("camera_matrix")
            dist_c = cam_conf.get("dist_coeffs")

            world_points = get_main_room_points(custom_rects)
            raw_px = cam_conf.get("pixel_points", [])
            pixel_points = [(int(p[0]), int(p[1])) for p in raw_px]

            view.update_scene(persons, room_dims, world_points, pixel_points,
                              camera_pos_label=cam_pos_label, custom_rectangles=custom_rects,
                              img_size=res, camera_matrix=cam_matrix, dist_coeffs=dist_c)

        now = time.time()

        # ==========================================
        # B.Master View
        # ==========================================
        if cam_name != "SYSTEM_REFRESH" and cam_name != "MASTER_FUSION":
            self.person_cache[cam_name] = persons

            # LIMITIERUNG: Master View maximal 30x pro Sekunde aktualisieren
        if now - getattr(self, 'last_master_render_time', 0) > 0.033:
            self.last_master_render_time = now
            all_persons_combined = []
            for p_list in self.person_cache.values():
                all_persons_combined.extend(p_list)

            master_conf = current_config.get("CAMERA_1", {})
            master_dims = master_conf.get("room_dimensions", {"width": 600, "height": 250, "depth": 800})
            master_rects = master_conf.get("custom_rectangles", [])
            master_world_points = get_main_room_points(master_rects)

            master_res = master_conf.get("resolution", [1920, 1080])
            master_cam_matrix = master_conf.get("camera_matrix")
            master_dist_c = master_conf.get("dist_coeffs")

            if self.view_master:
                self.view_master.update_scene(
                    all_persons_combined, master_dims,
                    camera_points_3d=master_world_points, pixel_points=[],
                    custom_rectangles=master_rects,
                    img_size=master_res,
                    camera_matrix=master_cam_matrix,
                    dist_coeffs=master_dist_c
                )

        # ==========================================
        # C. Live Metrik
        # ==========================================
        if now - getattr(self, 'last_details_render_time', 0) > 0.1:
            self.last_details_render_time = now
            if self.details_window and self.details_window.isVisible() and self.controller:
                self.details_window.update_data(self.controller.tracker.global_persons)

    def __on_register_client(self, name: str, ip: str) -> None:
        ts = self.__get_timestamp()
        if name in self.__client_items:
            item = self.__client_items[name]
            item.setText(2, "Active")
            item.setText(3, ts)
            item.setForeground(2, Qt.GlobalColor.darkGreen)
        else:
            item = QTreeWidgetItem([name, ip, "Active", ts])
            item.setForeground(2, Qt.GlobalColor.darkGreen)
            self.tree_clients.addTopLevelItem(item)
            self.__client_items[name] = item
            self.__on_log_message(f"Neu registriert: {name}")

    def __on_update_heartbeat(self, name: str) -> None:
        """Aktualisiert den Status eines Clients auf "Active" und aktualisiert den Ping-Zeitstempel. Wird aufgerufen, wenn ein Heartbeat von einem Client empfangen wird."""
        if name in self.__client_items:
            item = self.__client_items[name]
            item.setText(2, "Active")
            item.setText(3, self.__get_timestamp())
            item.setForeground(2, Qt.GlobalColor.darkGreen)

    def __on_set_offline(self, name: str) -> None:
        if name in self.__client_items:
            item = self.__client_items[name]
            item.setText(2, "OFFLINE")
            item.setForeground(2, Qt.GlobalColor.red)
            self.__on_log_message(f"Client verloren: {name}")

            if name in self.person_cache:
                self.person_cache[name] = []
                self.__on_update_camera_view("SYSTEM_REFRESH", [])

    def __on_log_message(self, message: str) -> None:
        ts = self.__get_timestamp()
        self.txt_log.append(f"[{ts}] {message}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def __get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")