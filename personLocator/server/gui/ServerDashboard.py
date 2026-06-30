import datetime
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Union
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QTextEdit, QGroupBox, QGridLayout, QHBoxLayout, QPushButton,
                             QMessageBox, QLabel, QCheckBox, QComboBox, QDialogButtonBox, QDialog, QRadioButton,
                             QFormLayout)
from PyQt6.QtCore import pyqtSignal, Qt, QEvent, QTimer

from server.gui.windows.PersonInfoWindow import PersonInfoWindow
from server.gui.views.Server3DView import Server3DView
from server.gui.windows.TrackingSettingsWindow import TrackingSettingsWindow
from server.gui.windows.EvaluationWindow import EvaluationWindow
from server.gui.windows.DatabaseViewerWindow import DatabaseViewerWindow
from server.gui.windows.MasterSettingsWindow import MasterSettingsWindow
from server.gui.windows.RoomObjectsWindow import RoomObjectsWindow
from server.config.ConfigManager import ConfigManager
from server.core.math.GeometryMath import GeometryMath
import time


class ServerSettingsWindow(QDialog):
    """Dialog für globale Performance- und Render-Einstellungen."""

    def __init__(self, current_mode: str, current_res_master: tuple[int, int], current_res_single: tuple[int, int], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚙️ Server & Performance Settings")
        self.resize(400, 300)
        layout: QVBoxLayout = QVBoxLayout(self)

        group_mode: QGroupBox = QGroupBox("Render-Modus")
        mode_layout: QVBoxLayout = QVBoxLayout(group_mode)

        self.rb_normal: QRadioButton = QRadioButton("🟢 Normal Mode (Alle GUIs & Kameras zeichnen)")
        self.rb_light: QRadioButton = QRadioButton("🟡 Performance Light (Nur Master View zeichnen)")
        self.rb_perf: QRadioButton = QRadioButton("🔴 Performance Mode (KEINE Grafik, nur Mathematik)")

        if current_mode == "performance":
            self.rb_perf.setChecked(True)
        elif current_mode == "performance_light":
            self.rb_light.setChecked(True)
        else:
            self.rb_normal.setChecked(True)

        mode_layout.addWidget(self.rb_normal)
        mode_layout.addWidget(self.rb_light)
        mode_layout.addWidget(self.rb_perf)
        layout.addWidget(group_mode)

        group_res: QGroupBox = QGroupBox("Render-Auflösungen")
        res_layout: QFormLayout = QFormLayout(group_res)

        self.combo_master: QComboBox = QComboBox()
        self.combo_master.addItems(["1920x1080 (Full HD)", "1280x720 (HD)", "800x450 (Low)"])
        if current_res_master[0] == 1920:
            self.combo_master.setCurrentIndex(0)
        elif current_res_master[0] == 1280:
            self.combo_master.setCurrentIndex(1)
        else:
            self.combo_master.setCurrentIndex(2)

        self.combo_single: QComboBox = QComboBox()
        self.combo_single.addItems(["1280x720 (HD)", "800x450 (Normal)", "640x360 (Low)", "480x270 (Potato)"])
        if current_res_single[0] == 1280:
            self.combo_single.setCurrentIndex(0)
        elif current_res_single[0] == 800:
            self.combo_single.setCurrentIndex(1)
        elif current_res_single[0] == 640:
            self.combo_single.setCurrentIndex(2)
        else:
            self.combo_single.setCurrentIndex(3)

        res_layout.addRow("Master View:", self.combo_master)
        res_layout.addRow("Einzelkameras:", self.combo_single)
        layout.addWidget(group_res)

        buttons: QDialogButtonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> tuple[str, tuple[int, int], tuple[int, int]]:
        mode: str = "normal"
        if self.rb_light.isChecked():
            mode = "performance_light"
        if self.rb_perf.isChecked():
            mode = "performance"
        
        res_m: tuple[int, int] = (1920, 1080) if self.combo_master.currentIndex() == 0 else (
            (1280, 720) if self.combo_master.currentIndex() == 1 else (800, 450))
        
        idx_s: int = self.combo_single.currentIndex()
        res_s: tuple[int, int] = (1280, 720) if idx_s == 0 else (
            (800, 450) if idx_s == 1 else ((640, 360) if idx_s == 2 else (480, 270)))
        
        return mode, res_m, res_s


class ServerDashboard(QMainWindow):
    sig_update_camera_view: pyqtSignal = pyqtSignal(str, list)
    sig_register_client: pyqtSignal = pyqtSignal(str, str)
    sig_update_heartbeat: pyqtSignal = pyqtSignal(str)
    sig_set_offline: pyqtSignal = pyqtSignal(str)
    sig_log_message: pyqtSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Server Dashboard - Master Control")
        self.resize(1600, 900)

        self.gui_render_mode = "normal"
        self.res_master = (1280, 720)
        self.res_single = (640, 360)
        self.current_tracking_mode = "hungarian"
        self.current_smoothing_filter = "one_euro"
        self.current_ik_mode = "fabrik"
        self.current_triangulation_mode = "wls"

        self.controller = None
        self.eval_window = None
        self.person_info_window = None
        self.camera_views = {}
        self.fullscreen_view = None
        self.view_master = None
        self.__client_items: Dict[str, QTreeWidgetItem] = {}
        self.person_cache = {}
        self.cache_timestamps = {}
        self.ray_settings = {}
        self.ray_checkboxes = {}

        self.master_config = ConfigManager.get_master_config_data()
        self._settings_loaded_from_db = False

        self.__setup_ui()
        self.__connect_signals()
        self.__initialize_static_views()

    def set_controller(self, controller_instance):
        self.controller = controller_instance
        if self.view_master: 
            self.view_master.renderer.controller = self.controller
            if hasattr(self.controller, 'view_master'):
                self.controller.view_master = self.view_master
        
        for view in self.camera_views.values(): 
            view.renderer.controller = self.controller

    def show_window(self) -> None:
        self.show()
        QTimer.singleShot(500, lambda: self.__on_update_camera_view("SYSTEM_REFRESH", []))

    def load_all_settings_from_db(self):
        if not self.controller or not hasattr(self.controller, 'system_db'): return
        try:
            saved = self.controller.system_db.get_master_settings()
            if saved:
                self.ray_settings.update(saved)
                for cam, chk in self.ray_checkboxes.items():
                    chk.blockSignals(True)
                    chk.setChecked(self.ray_settings.get(cam, {}).get("show_single_rays", True))
                    chk.blockSignals(False)

            srv = self.controller.system_db.get_server_settings()
            if srv:
                self.gui_render_mode = srv.get("mode", "normal")
                self.res_master = tuple(srv.get("res_master", [1280, 720]))
                self.res_single = tuple(srv.get("res_single", [640, 360]))
                self.grid_group.setVisible(self.gui_render_mode == "normal")

            trk = self.controller.system_db.get_tracking_settings()
            if trk:
                self.current_tracking_mode = trk.get("tracking_mode", "hungarian")
                self.current_smoothing_filter = trk.get("smoothing_mode", "one_euro")
                self.current_ik_mode = trk.get("ik_mode", "fabrik")
                self.current_triangulation_mode = trk.get("triangulation_mode", "wls")

                all_views = [self.view_master] + list(self.camera_views.values())
                for v in all_views:
                    if v and hasattr(v, 'renderer'):
                        v.renderer.tracking_mode = self.current_tracking_mode
                        v.renderer.smoothing_mode = self.current_smoothing_filter
                        v.renderer.ik_mode = self.current_ik_mode
                        v.renderer.triangulation_mode = self.current_triangulation_mode

            self.log_message("✅ Alle persistenten Einstellungen aus DB geladen.")
            self._settings_loaded_from_db = True
        except Exception as e:
            self.log_message(f"⚠️ DB-Ladefehler: {e}")

    def __setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.master_group = QGroupBox("MASTER VIEW (Gesamt)")
        m_lay = QVBoxLayout(self.master_group)
        self.view_master = Server3DView("GESAMTANSICHT", is_master=True)
        self.view_master.installEventFilter(self)
        m_lay.addWidget(self.view_master)

        self.master_control_layout = QHBoxLayout()
        self.lbl_fusion_mode = QLabel("Warte auf Daten...")
        self.master_control_layout.addWidget(self.lbl_fusion_mode)
        self.ray_buttons_layout = QHBoxLayout()
        self.master_control_layout.addLayout(self.ray_buttons_layout)
        m_lay.addLayout(self.master_control_layout)
        left_layout.addWidget(self.master_group, 5)

        self.grid_group = QGroupBox("Einzelkameras")
        self.grid_layout = QGridLayout(self.grid_group)
        left_layout.addWidget(self.grid_group, 4)
        main_layout.addWidget(left_widget, 4)

        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)

        tree_grp = QGroupBox("Clients")
        t_lay = QVBoxLayout(tree_grp)
        self.tree_clients = QTreeWidget()
        self.tree_clients.setHeaderLabels(["Client", "IP", "Status", "Ping"])
        t_lay.addWidget(self.tree_clients)
        right_layout.addWidget(tree_grp, 1)

        btns = [
            ("⚙️ Filter & Fusion", self.open_tracking_settings_window, "#1a4d2e"),
            ("👤 Personen Monitor", self.open_person_info_window, "#0055AA"),
            ("📈 System Evaluierung", self.open_evaluation_window, "#5500AA"),
            ("🗄️ Datenbank Studio", self.open_database_viewer, "#444444"),
            ("⚙️ Server Settings", self.open_server_settings, "#555555"),
            ("⚙️ Master View Settings", self.open_master_settings, "#008080"),
            ("📦 Raum-Objekte", self.open_room_objects_window, "#b07000"),
            ("🎯 Pointer-Monitor", self.open_pointers_window, "#C71585"),
            ("📷 Kamera 3D-Positionen", self.show_camera_positions, "#6a0dad"),
            ("🔍 SQL-Inspektor", self.toggle_sql_logging, "#2b7a0b")
        ]
        for txt, method, col in btns:
            b = QPushButton(txt)
            b.setStyleSheet(f"background-color: {col}; color: white; padding: 8px; font-weight: bold;")
            b.clicked.connect(method)
            right_layout.addWidget(b)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_grp = QGroupBox("Events")
        right_layout.addWidget(log_grp, 1)
        log_grp_lay = QVBoxLayout(log_grp)
        log_grp_lay.addWidget(self.txt_log)

        self.right_widget.setFixedWidth(350)
        main_layout.addWidget(self.right_widget, 1)

    def open_person_info_window(self):
        if not self.person_info_window:
            self.person_info_window = PersonInfoWindow(self)
        self.person_info_window.show()

    def open_server_settings(self):
        dlg = ServerSettingsWindow(self.gui_render_mode, self.res_master, self.res_single, self)
        if dlg.exec():
            m, rm, rs = dlg.get_settings()
            self.gui_render_mode, self.res_master, self.res_single = m, rm, rs
            self.grid_group.setVisible(m == "normal")
            if self.controller: self.controller.system_db.update_server_settings(
                {"mode": m, "res_master": rm, "res_single": rs})

    def open_tracking_settings_window(self):
        dlg = TrackingSettingsWindow(self.current_tracking_mode, self)
        if dlg.exec():
            self.current_tracking_mode = dlg.get_selected_mode()
            self.current_smoothing_filter = dlg.get_selected_filter()
            self.current_ik_mode = dlg.get_selected_ik()
            self.current_triangulation_mode = dlg.get_selected_triangulation()

            all_views = [self.view_master] + list(self.camera_views.values())
            for v in all_views:
                if v and hasattr(v, 'renderer'):
                    v.renderer.tracking_mode = self.current_tracking_mode
                    v.renderer.smoothing_mode = self.current_smoothing_filter
                    v.renderer.ik_mode = self.current_ik_mode
                    v.renderer.triangulation_mode = self.current_triangulation_mode

            if self.controller: self.controller.system_db.update_tracking_settings({
                "tracking_mode": self.current_tracking_mode, "smoothing_mode": self.current_smoothing_filter,
                "ik_mode": self.current_ik_mode, "triangulation_mode": self.current_triangulation_mode
            })

    def open_evaluation_window(self):
        if not self.controller: return
        if not self.eval_window: self.eval_window = EvaluationWindow(self.controller, self)
        self.eval_window.show()

    def open_database_viewer(self):
        if not self.controller: return
        if not hasattr(self.controller, 'db_viewer') or not self.controller.db_viewer:
            self.controller.db_viewer = DatabaseViewerWindow(self.controller, self)
        self.controller.db_viewer.show()

    def open_master_settings(self):
        if not self.controller: return
        cams = [c for c in self.controller.config_cache.keys() if c.startswith("CAMERA_")] + ["MASTER_FUSION"]
        dlg = MasterSettingsWindow(self.ray_settings, cams, self)
        if dlg.exec():
            self.ray_settings = dlg.get_settings()
            if self.controller: self.controller.system_db.update_master_settings(self.ray_settings)

    def open_room_objects_window(self):
        if not self.controller:
            QMessageBox.information(self, "Raum-Objekte", "Server-Controller nicht verfügbar.")
            return
        dlg = RoomObjectsWindow(self.controller, self)
        if dlg.exec():
            self.log_message(f"📦 Raum-Objekte aktualisiert ({len(self.controller.room_objects)} Objekte).")
            if self.view_master and self.view_master.last_scene_data is not None:
                self.view_master.last_scene_data["room_objects"] = list(self.controller.room_objects or [])
                self.view_master._needs_render = True
            self.last_master_render_time = 0.0

    def open_pointers_window(self):
        """Öffnet den Pointer-Monitor (Zeigt Interaktionen)."""
        if not self.controller:
            QMessageBox.information(self, "Pointer-Monitor", "Server-Controller nicht verfügbar.")
            return

        from server.gui.windows.PointersWindow import PointersWindow
        
        if not hasattr(self, "pointers_window") or self.pointers_window is None:
            self.pointers_window = PointersWindow(self.controller, self)
        
        self.pointers_window.show()
        self.pointers_window.raise_()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if source == self.view_master or source in self.camera_views.values():
                self.toggle_fullscreen(source)
                return True
        return super().eventFilter(source, event)

    def toggle_fullscreen(self, view):
        if self.fullscreen_view is None:
            self.fullscreen_view = view
            self.right_widget.setVisible(False)
            if view == self.view_master:
                self.grid_group.setVisible(False)
            else:
                self.master_group.setVisible(False)
                for v in self.camera_views.values():
                    if v != view: v.setVisible(False)
        else:
            self.fullscreen_view = None
            self.right_widget.setVisible(True)
            self.master_group.setVisible(True)
            self.grid_group.setVisible(True)
            for v in self.camera_views.values(): v.setVisible(True)

    def _ensure_camera_view(self, cam_name: str):
        """Erstellt ein neues Kamerabild und ordnet das Grid sicher neu an."""
        if cam_name not in self.camera_views:
            import math
            v = Server3DView(cam_name, is_master=False)
            if self.controller: v.renderer.controller = self.controller
            v.renderer.tracking_mode = self.current_tracking_mode
            v.renderer.smoothing_mode = self.current_smoothing_filter
            self.camera_views[cam_name] = v

            while self.grid_layout.count():
                item = self.grid_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

            sorted_keys = sorted(self.camera_views.keys())
            cols = math.ceil(math.sqrt(len(sorted_keys)))
            if cols == 0: cols = 1

            for i, k in enumerate(sorted_keys):
                self.grid_layout.addWidget(self.camera_views[k], i // cols, i % cols)
                self.camera_views[k].show()

    def log_message(self, msg):
        try:
            self.sig_log_message.emit(msg)
        except RuntimeError:
            pass

    def __on_log_message(self, msg):
        self.txt_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def __connect_signals(self):
        self.sig_log_message.connect(self.__on_log_message)
        self.sig_update_camera_view.connect(self.__on_update_camera_view)
        self.sig_register_client.connect(self.__on_register_client)
        self.sig_update_heartbeat.connect(self.__on_update_heartbeat)
        self.sig_set_offline.connect(self.__on_set_offline)

    def __initialize_static_views(self):
        cam_names = [k for k in self.master_config.keys() if k.startswith("CAMERA_")]
        for cam_name in cam_names:
            self._ensure_camera_view(cam_name)

    def show_camera_positions(self):
        if not self.controller or not self.controller.config_cache:
            QMessageBox.information(self, "Kamera Positionen", "Noch keine Kameras verbunden.")
            return

        info_text = "📍 BERECHNETE KAMERA-POSITIONEN (Im 3D-Raum)\n" + "=" * 65 + "\n\n"
        found_any = False

        for cam_name, cam_conf in list(self.controller.config_cache.items()):
            if not cam_name.startswith("CAMERA"): continue
            custom_rects = cam_conf.get("custom_rectangles", [])
            res = cam_conf.get("resolution") or [1920, 1080]
            cam_matrix = cam_conf.get("camera_matrix")
            dist_c = cam_conf.get("dist_coeffs")

            pose = GeometryMath.get_camera_pose([], [], custom_rectangles=custom_rects,
                                                img_size=res, camera_matrix_override=cam_matrix,
                                                dist_coeffs_override=dist_c, camera_name=cam_name)
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
                info_text += f"   Winkel      : Pitch: {deg[0]:5.1f}° | Yaw: {deg[1]:5.1f}° | Roll: {deg[2]:5.1f}°\n"
                info_text += "-" * 65 + "\n"

        msg = QMessageBox(self)
        msg.setWindowTitle("Kamera 3D Positionen")
        msg.setText(info_text if found_any else "Keine Kameras kalibriert.")
        msg.setStyleSheet("QLabel{font-family: Consolas; font-size: 14px;}")
        msg.exec()

    def toggle_sql_logging(self):
        if not self.controller or not hasattr(self.controller, 'system_db'): return
        is_active = self.controller.system_db.toggle_sql_logging()
        self.log_message(f"SQL-Inspektor wurde {'AKTIVIERT' if is_active else 'DEAKTIVIERT'}.")

    def register_client(self, name: str, ip: str) -> None:
        self.sig_register_client.emit(name, ip)

    def update_heartbeat(self, name: str) -> None:
        self.sig_update_heartbeat.emit(name)

    def set_client_offline(self, name):
        try:
            self.sig_set_offline.emit(name)
        except RuntimeError:
            pass

    def update_camera_data(self, camera_name: str, person_list: list):
        try:
            self.sig_update_camera_view.emit(camera_name, person_list)
        except RuntimeError:
            pass

    def closeEvent(self, event):
        try:
            if self.view_master and hasattr(self.view_master, 'render_worker'):
                self.view_master.render_worker.stop()
            for view in self.camera_views.values():
                if hasattr(view, 'render_worker'): view.render_worker.stop()
            if self.controller and hasattr(self.controller, 'stop'):
                self.controller.stop()
        except Exception as e:
            print(f"Fehler beim sauberen Herunterfahren: {e}")
        event.accept()

    def __on_register_client(self, name: str, ip: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
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
        if name in self.__client_items:
            item = self.__client_items[name]
            item.setText(2, "Active")
            item.setText(3, datetime.datetime.now().strftime("%H:%M:%S"))
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

    def __on_update_camera_view(self, cam_name: str, persons: list):
        if not self.controller: return
        if not self._settings_loaded_from_db and hasattr(self.controller, 'system_db'):
            self.load_all_settings_from_db()

        if cam_name.startswith("CAMERA_"): self._ensure_camera_view(cam_name)
        if not hasattr(self, 'cache_timestamps'): self.cache_timestamps = {}

        current_config = self.controller.config_cache

        def get_main_room_points(custom_rects):
            point_labels = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]
            ordered_pts = []
            for rect in custom_rects:
                if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("type") == "Haupt-Kalibrierung":
                    corners = rect.get("corners", [])
                    for lbl in point_labels:
                        found = next((c for c in corners if c.get("label") == lbl), None)
                        if found:
                            ordered_pts.append(np.array([found["x"], found["y"], found["z"]], dtype=np.float32))
            return ordered_pts

        now = time.time()

        if cam_name in self.camera_views:
            for p in persons: p["cam_name"] = cam_name
            if not hasattr(self, 'last_single_render_times'): self.last_single_render_times = {}

            if now - self.last_single_render_times.get(cam_name, 0) > 0.033:
                self.last_single_render_times[cam_name] = now
                view = self.camera_views[cam_name]
                cam_conf = current_config.get(cam_name, {})
                room_dims = cam_conf.get("room_dimensions", {"width": 600, "height": 250, "depth": 800})
                cam_pos_label = cam_conf.get("position", None)
                custom_rects = cam_conf.get("custom_rectangles", [])
                world_points = get_main_room_points(custom_rects)
                pixel_points = [(int(p[0]), int(p[1])) for p in cam_conf.get("pixel_points", [])]

                if self.gui_render_mode not in ["performance_light", "performance"]:
                    safe_res = cam_conf.get("resolution") or [1920, 1080]
                    view.update_scene(persons, room_dims, world_points, pixel_points,
                                      camera_pos_label=cam_pos_label, custom_rectangles=custom_rects,
                                      img_size=safe_res, camera_matrix=cam_conf.get("camera_matrix"),
                                      dist_coeffs=cam_conf.get("dist_coeffs"), active_ray_cameras=self.ray_settings,
                                      render_graphics=True, custom_res=self.res_single)

        if cam_name.startswith("CAMERA_"):
            self.person_cache[cam_name] = persons
            self.cache_timestamps[cam_name] = now

        if now - getattr(self, 'last_master_render_time', 0) > 0.033:
            self.last_master_render_time = now
            force_single = self.ray_settings.get("force_single_person", False)

            all_raw_persons = []
            active_cams_now = 0

            for c_name, p_list in self.person_cache.items():
                if c_name.startswith("CAMERA_"):
                    if now - self.cache_timestamps.get(c_name, 0) > 0.5: continue
                    active_cams_now += 1
                    for p in p_list:
                        p_raw = dict(p)
                        p_raw["cam_name"] = c_name
                        if force_single:
                            p_raw["id"] = 1
                            p_raw["status"] = "FORCE MERGE (ID 1)"
                        all_raw_persons.append(p_raw)

            if active_cams_now >= 2:
                self.lbl_fusion_mode.setText(f"🚀 Modus: OPTISCHE TRIANGULATION ({active_cams_now} Kameras)")
                self.lbl_fusion_mode.setStyleSheet(
                    "color: #00FF96; font-size: 14px; font-weight: bold; margin-right: 20px;")
            elif active_cams_now == 1:
                self.lbl_fusion_mode.setText("🎯 Modus: SINGLE-RAYCASTING (Warte auf 2. Kamera...)")
                self.lbl_fusion_mode.setStyleSheet(
                    "color: #00FFFF; font-size: 14px; font-weight: bold; margin-right: 20px;")
            else:
                self.lbl_fusion_mode.setText("Warte auf Kamera-Daten...")
                self.lbl_fusion_mode.setStyleSheet(
                    "color: #AAA; font-size: 14px; font-weight: bold; margin-right: 20px;")

            for c_name in list(current_config.keys()):
                if c_name.startswith("CAMERA_") and c_name not in self.ray_checkboxes:
                    chk = QCheckBox(f"Rays {c_name.replace('CAMERA_', '')}")
                    chk.blockSignals(True)
                    is_active = self.ray_settings.get(c_name, {}).get("show_single_rays", True)
                    chk.setChecked(is_active)
                    chk.setStyleSheet(
                        "color: white; background: #333; padding: 5px; border-radius: 3px; margin-right: 5px; font-weight: bold;")
                    chk.stateChanged.connect(
                        lambda state, name=c_name: self._toggle_rays(name, state == Qt.CheckState.Checked.value))
                    self.ray_buttons_layout.addWidget(chk)
                    self.ray_checkboxes[c_name] = chk
                    chk.blockSignals(False)

            all_rects_dict = {}
            for c_name, c_conf in current_config.items():
                if c_name.startswith("CAMERA_"):
                    for rect in c_conf.get("custom_rectangles", []):
                        if "internal_id" in rect:
                            all_rects_dict[rect["internal_id"]] = rect

            master_rects = list(all_rects_dict.values())
            master_world_points = get_main_room_points(master_rects)

            master_conf = current_config.get("CAMERA_1", {})
            if not master_conf:
                for c_name, c_conf in current_config.items():
                    if c_name.startswith("CAMERA_"):
                        master_conf = c_conf
                        break

            master_dims = master_conf.get("room_dimensions", {"width": 600, "height": 250, "depth": 800})

            all_cameras_3d = {}
            if hasattr(self.controller, 'config_cache'):
                for name, cfg in list(self.controller.config_cache.items()):
                    if name == "Camera_ALL": continue
                    cam_3d_data = cfg.get("cam_3d_data")
                    if cam_3d_data: all_cameras_3d[name] = cam_3d_data

            if self.view_master:
                do_render = (self.gui_render_mode != "performance")
                safe_master_res = master_conf.get("resolution") or [1920, 1080]

                master_room_objects: list[dict[str, Any]] = list(getattr(self.controller, "room_objects", []) or [])

                self.view_master.update_scene(
                    all_raw_persons, master_dims,
                    camera_points_3d=master_world_points, pixel_points=[],
                    custom_rectangles=master_rects,
                    img_size=safe_master_res,
                    camera_matrix=master_conf.get("camera_matrix"),
                    dist_coeffs=master_conf.get("dist_coeffs"),
                    all_cameras_3d=all_cameras_3d,
                    active_ray_cameras=self.ray_settings,
                    render_graphics=do_render,
                    custom_res=self.res_master,
                    room_objects=master_room_objects
                )

        if now - getattr(self, 'last_details_render_time', 0) > 0.1:
            self.last_details_render_time = now
            if self.person_info_window and self.person_info_window.isVisible() and self.controller:
                try:
                    self.person_info_window.update_data(self.controller.tracker.global_persons)
                except Exception as e:
                    import traceback
                    print(f"KRITISCHER FEHLER im Personen-Monitor: {e}")
                    traceback.print_exc()

    def _toggle_rays(self, cam_name: str, is_active: bool):
        if cam_name not in self.ray_settings:
            self.ray_settings[cam_name] = {"master": True, "show_rays": True, "show_single_rays": True,
                                           "show_points": True, "show_bones": False,
                                           "joints": {i: True for i in range(17)}}

        self.ray_settings[cam_name]["show_single_rays"] = is_active
        self.log_message(f"Einzelansicht {cam_name}: Strahlen {'AN' if is_active else 'AUS'}")
        if self.controller and hasattr(self.controller, 'system_db'):
            self.controller.system_db.update_master_settings(self.ray_settings)