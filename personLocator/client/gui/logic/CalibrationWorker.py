import logging
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from client.gui.logic.CalibrationToolbox import CalibrationToolbox
from client.gui.logic.label.render.RendererDashboard import RendererDashboard
from client.gui.logic.label.utilityClasses.utilityClassCalibration.Worker.FrameProcessor import FrameProcessor
from client.logic.sensors.AsyncDetector import AsyncDetector
from client.logic.sensors.PersonDetector import PersonDetector
from client.logic.sensors.CameraSource import CameraSource
from client.utils.ConfigManager import ConfigManager


class CalibrationWorker(QThread):
    image_data = pyqtSignal(QImage, QImage)
    points_data = pyqtSignal(list)
    measurement_data = pyqtSignal(object)
    full_res_image_data = pyqtSignal(QImage)

    def __init__(self, camera_name: str):
        super().__init__()
        self.camera_name = camera_name
        self.prev_frame_time = 0
        self.current_fps = 0
        self._is_running = True
        self.settings = {}

        # Komponenten
        self.toolbox = CalibrationToolbox(camera_name)
        self.detector = PersonDetector()

        self.__setup_hardware_config()

        self.camera = CameraSource(
            index=self.settings.get("index", 0),
            resolution=self.settings.get("res", "Auto")
        )

        # AI Thread
        self.ai_thread = AsyncDetector(camera_name, self.detector, self.toolbox)
        self.ai_thread.results_ready.connect(self._on_ai_results)
        self.emit_full_res = False
        self.last_person_results = []
        self._needs_camera_reload = False

    def set_precision_mode(self, active: bool):
        self.emit_full_res = active

    def run(self):
        """Hauptschleife."""
        if not self.camera.open():
            logging.error("Konnte Kamera nicht öffnen!")
            return

        self.ai_thread.start()
        target_fps = self.settings.get("target_fps", 30)
        sleep_time = int(1000 / target_fps) if target_fps > 0 else 33

        while self._is_running:
            if self._needs_camera_reload:
                if hasattr(self, 'ai_thread'):
                    self.ai_thread.reset_tracking()
                self.camera.release()
                self.camera = CameraSource(index=self.settings["index"], resolution=self.settings["res"])
                self.camera.open()
                self._needs_camera_reload = False
            start_time = time.time()

            # fps Einstellung mit gleitendem Durchschnitt, damit sie nicht zu sehr schwankt
            time_diff = start_time - self.prev_frame_time
            if time_diff > 0:
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1 / time_diff)
            self.prev_frame_time = start_time

            # 1. Bild holen
            ret, frame = self.camera.read()

            if ret and frame is not None:
                # 2. Verarbeiten
                frame = self._preprocess_frame(frame)

                # 3. AI Füttern
                if not self.emit_full_res:
                    self.ai_thread.update_frame(frame)

                # 4. GUI Update
                self._update_ui_signals(frame)

            process_time_ms = (time.time() - start_time) * 1000
            wait_time = max(1, int(sleep_time - process_time_ms))
            self.msleep(wait_time)

        self.camera.release()
        self.ai_thread.stop()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Wendet die aktuellen Zoom- und Rotations-Einstellungen auf das Kamerabild an."""
        frame = FrameProcessor.apply_zoom(frame, self.settings.get("zoom", 1.0))
        return FrameProcessor.apply_rotation(frame, self.settings.get("rotation", 0))


    def add_zone(self, mode: str, points: list):
        if mode == "dead":
            self.toolbox.zone_manager.add_dead_zone(points)
        elif mode == "mirror":
            self.toolbox.zone_manager.add_mirror_zone(points)

    def delete_last_zone(self):
        try:
            self.toolbox.zone_manager.delete_last_zone("dead")
            self.toolbox.zone_manager.delete_last_zone("mirror")
        except Exception as e:
            logging.error(f"Konnte letzte Zone nicht löschen: {e}")

    def clear_zones(self):
        self.toolbox.zone_manager.clear_zones()

    def _on_ai_results(self, detections, person_results):
        """Wird aufgerufen, wenn der AI-Thread neue Ergebnisse hat. Aktualisiert die internen Daten und feuert die entsprechenden Signale."""
        self.last_person_results = person_results

        # Datenpaket senden
        analysis = self.toolbox.get_geometric_analysis()
        self.measurement_data.emit({
            "persons": person_results,
            "geometry": analysis,
            "fps": round(self.current_fps, 1)
        })

    def _update_ui_signals(self, frame: np.ndarray):
        """Feuert die Signale für die GUI-Updates. Je nach Modus entweder mit oder ohne KI-Overlay."""
        if self.emit_full_res:
            if getattr(self, 'lens_window_ready', True):
                self.lens_window_ready = False
                self.full_res_image_data.emit(self._to_qimage(frame))
            return

        # --- NORMALER MODUS (Mit KI und 3D) ---
        rendering_active = self.toolbox.view_options.get("render_3d_enabled", True)
        h, w = frame.shape[:2]
        scale = min(1.0, 960 / w)

        # 2D Kamerabild vorbereiten
        vis_frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        self.detector.draw_detections(vis_frame, self.last_person_results, scale=scale)
        ui_frame = self.toolbox.draw_toolbox_ui(vis_frame, scale=scale)

        if rendering_active:
            # 1. 3D-Raum in hoher Auflösung rendern
            preview_3d = self.toolbox.generate_3d_preview(self.last_person_results)

            # 2. 3D-Ansicht auf die GUI-Breite skalieren
            target_w = 1000
            h_p, w_p = preview_3d.shape[:2]
            scale_p = target_w / w_p
            ui_3d_part = cv2.resize(preview_3d, (target_w, int(h_p * scale_p)), interpolation=cv2.INTER_AREA)

            # 3. Den Graphen direkt über die Toolbox-Werte rendern
            distortion_graph = RendererDashboard.render_live_distortion_graph(
                pixel_points=self.toolbox.pixel_points,
                world_points=self.toolbox.world_points_3d,
                custom_rectangles=self.toolbox.custom_rectangles,
                room_dims=self.toolbox.real_room_dims,
                person_results=self.last_person_results,
                width=target_w,
                height=220,
                yaw_deg=self.toolbox.rotation_yaw,
                pitch_deg=self.toolbox.rotation_pitch
            )

            # 4. Stapeln
            ui_3d = np.vstack((ui_3d_part, distortion_graph))
        else:
            ui_3d = np.zeros((750, 1000, 3), dtype=np.uint8)

        self.image_data.emit(self._to_qimage(ui_frame), self._to_qimage(ui_3d))

    def _to_qimage(self, arr: np.ndarray) -> QImage:
        """Konvertiert ein OpenCV-BGR-Array in ein QImage. Stellt sicher, dass das Array C-kontig ist, um Fehler zu vermeiden."""
        if arr is None: return QImage()
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        image = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        return image.copy()



    def trigger_config_reload(self) -> None:
        """Wird aufgerufen, wenn in der GUI Einstellungen geändert wurden."""
        config = ConfigManager.get_camera_settings(self.camera_name)

        old_index = self.settings.get("index")
        new_index = config.get("camera_index", 0)
        old_res = self.settings.get("res")
        new_res = config.get("resolution", [1920, 1080])

        if old_index != new_index or old_res != new_res:
            logging.info(f"Kamera-Hardware Update angefordert: {old_res} -> {new_res}")
            self.settings.update({"index": new_index, "res": new_res})
            self._needs_camera_reload = True

        self.settings.update({
            "zoom": config.get("zoom", 1.0),
            "rotation": config.get("rotation", 0),
            "target_fps": config.get("target_fps", 30)
        })

        self.toolbox._load_config()

    def __setup_hardware_config(self):
        """Lädt NUR die Settings, die der Worker für die Kamera-Hardware braucht."""
        config = ConfigManager.get_camera_settings(self.camera_name)
        self.settings = {
            "index": config.get("camera_index", 0),
            "zoom": config.get("zoom", 1.0),
            "rotation": config.get("rotation", 0),
            "res": config.get("resolution", "Auto"),
            "target_fps": config.get("target_fps", 30)
        }


    def set_person_height(self, p_id, h):
        if hasattr(self.toolbox, 'person_manager'):
            self.toolbox.person_manager.set_manual_height(p_id, h)

    def update_zoom(self, delta: int):
        self.toolbox.update_zoom(delta)

    def update_rotation(self, dx, dy):
        self.toolbox.update_rotation(dx, dy)

    def stop(self):
        self._is_running = False
        if hasattr(self, 'ai_thread'):
            self.ai_thread.stop()