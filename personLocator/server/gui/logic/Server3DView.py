import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtCore import Qt, QPoint, QTimer

from server.gui.logic.CalibrationRenderer import CalibrationRenderer


class Server3DView(QLabel):
    """
    Interaktiver 3D-Viewer für den Server.
    Nutzt QTimer für hochperformantes Entkoppeln von Daten-Empfang und Rendering!
    """

    def __init__(self, title="Kamera"):
        super().__init__()
        self.title = title
        self.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

        self.rotation_yaw = -30.0
        self.rotation_pitch = -20.0
        self.zoom_level = 1.0
        self.last_mouse_pos = QPoint()
        self.mouse_sensitivity = 0.5

        self.last_scene_data = None
        self._current_qimage = None

        self._needs_render = False
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._process_render_queue)
        self.render_timer.start(33)  # ca. 30 FPS Maximal-Limit für  GUI

    def update_scene(self, person_data_list, room_dims, camera_points_3d, pixel_points=None, camera_pos_label=None,
                     custom_rectangles=None, img_size=(1920, 1080), camera_matrix=None, dist_coeffs=None):
        """Wird von außen gerufen, um neue Daten zu übergeben. Es werden nur die Daten gespeichert, das Rendering passiert im Timer-Loop."""

        if person_data_list:
            for p in person_data_list:
                if 'pos' not in p and 'x' in p:
                    p['pos'] = np.array([p['x'], p['y'], p['z']], dtype=np.float32)
                elif 'pos' in p and not isinstance(p['pos'], np.ndarray):
                    p['pos'] = np.array(p['pos'], dtype=np.float32)

        if pixel_points is None: pixel_points = []

        self.last_scene_data = {
            "persons": person_data_list,
            "room": room_dims,
            "points_3d": camera_points_3d,
            "points_2d": pixel_points,
            "cam_pos": camera_pos_label,
            "custom_rects": custom_rectangles or [],
            "img_size": img_size,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs
        }

        self._needs_render = True

    # ==========================================
    # 2. Rendering (Gesteuert vom Timer)
    # ==========================================
    def _process_render_queue(self):
        """Wird exakt 30x pro Sekunde gerufen und rendert nur bei Bedarf."""
        if self._needs_render:
            self._render_now()
            self._needs_render = False

    def _render_now(self):
        """Rendert die aktuelle Szene basierend auf den zuletzt übergebenen Daten.
        Alle komplexen Berechnungen und das Zeichnen passieren hier,
        damit die update_scene-Methode so leichtgewichtig wie möglich bleibt."""
        if not self.last_scene_data: return
        d = self.last_scene_data

        MESH_CONNECTIONS = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (6, 7), (7, 5), (5, 4), (0, 4), (1, 6), (2, 5),
                            (3, 7)]
        POINT_LABELS = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]

        view_options = {
            "show_real_world": True, "show_camera_world": True,
            "show_skeleton_3d": True, "render_3d_enabled": True
        }

        try:
            rendered_img = CalibrationRenderer.render_3d_scene(
                world_points_3d=d["points_3d"],
                rotation_yaw=self.rotation_yaw,
                rotation_pitch=self.rotation_pitch,
                mesh_connections=MESH_CONNECTIONS,
                angles_to_measure=[],
                point_labels=POINT_LABELS,
                person_results=d["persons"],
                room_dims=d["room"],
                view_options=view_options,
                zoom_level=self.zoom_level,
                pixel_points=d["points_2d"],
                camera_pos_label=d.get("cam_pos"),
                custom_rectangles=d.get("custom_rects", []),
                img_size=d.get("img_size", (1920, 1080)),
                camera_matrix=d.get("camera_matrix"),
                dist_coeffs=d.get("dist_coeffs")
            )

            if rendered_img is not None:
                h, w, ch = rendered_img.shape
                self._current_qimage = QImage(rendered_img.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
                self.update()
        except Exception as e:
            print(f"Render Error: {e}")

    # ==========================================
    # 3. PAINT EVENT & MAUS
    # ==========================================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17, 17, 17))

        if self._current_qimage and not self._current_qimage.isNull():
            target_rect = self.rect()
            scaled_img = self._current_qimage.scaled(
                target_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
            )
            x = (target_rect.width() - scaled_img.width()) // 2
            y = (target_rect.height() - scaled_img.height()) // 2
            painter.drawImage(x, y, scaled_img)
        else:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"Warte auf {self.title}...")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            self.rotation_yaw += dx * self.mouse_sensitivity
            self.rotation_pitch = max(-89.0, min(89.0, self.rotation_pitch - (dy * self.mouse_sensitivity)))
            self.last_mouse_pos = event.pos()
            self._needs_render = True

    def wheelEvent(self, event):
        self.zoom_level = max(0.1, min(5.0, self.zoom_level + ((event.angleDelta().y() / 120.0) * 0.1)))
        self._needs_render = True