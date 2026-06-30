import time
import os
import platform
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QImage, QPainter, QColor, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from server.rendering.CalibrationRenderer import CalibrationRenderer

from server.core.math.GeometryMath import GeometryMath


def get_current_cpu_core() -> int:
    try:
        if hasattr(os, 'sched_getcpu'):
            return int(os.sched_getcpu())
        elif platform.system() == "Windows":
            import ctypes
            return int(ctypes.windll.kernel32.GetCurrentProcessorNumber())
    except Exception:
        pass
    return -1


class RenderWorker(QThread):
    frame_ready: pyqtSignal = pyqtSignal(object, int)

    def __init__(self, renderer: CalibrationRenderer) -> None:
        super().__init__()
        self.renderer: CalibrationRenderer = renderer
        self.current_data: Optional[dict[str, Any]] = None
        self.is_running: bool = True
        self.has_new_data: bool = False

    def run(self) -> None:
        while self.is_running:
            if self.has_new_data and self.current_data is not None:
                d: dict[str, Any] = self.current_data
                self.has_new_data = False

                try:
                    MESH_CONNECTIONS: list[tuple[int, int]] = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (6, 7), (7, 5), (5, 4), (0, 4), (1, 6),
                                        (2, 5), (3, 7)]
                    POINT_LABELS: list[str] = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]

                    view_options: dict[str, bool] = {
                        "show_real_world": True, "show_camera_world": True,
                        "show_skeleton_3d": True, "render_3d_enabled": True
                    }

                    raw_matrix: Optional[np.ndarray] = d.get("camera_matrix")
                    current_res: tuple[int, int] = d.get("img_size", (1920, 1080))

                    calib_res: list[int] = [1920, 1080]

                    scaled_matrix: Optional[np.ndarray] = GeometryMath.scale_camera_matrix(raw_matrix, calib_res, current_res)

                    rendered_img: Optional[np.ndarray] = self.renderer.render_3d_scene(
                        world_points_3d=d["points_3d"],
                        rotation_yaw=float(d["rotation_yaw"]),
                        rotation_pitch=float(d["rotation_pitch"]),
                        mesh_connections=MESH_CONNECTIONS,
                        angles_to_measure=[],
                        point_labels=POINT_LABELS,
                        person_results=d["persons"],
                        room_dims=d["room"],
                        view_options=view_options,
                        zoom_level=float(d["zoom_level"]),
                        pixel_points=d["points_2d"],
                        camera_pos_label=d.get("cam_pos"),
                        custom_rectangles=d.get("custom_rects", []),
                        img_size=current_res,
                        camera_matrix=scaled_matrix,
                        dist_coeffs=d.get("dist_coeffs"),
                        all_cameras_3d=d.get("all_cameras_3d", {}),
                        active_ray_cameras=d.get("active_ray_cameras", {}),
                        render_graphics=bool(d.get("render_graphics", True)),
                        custom_res=d.get("custom_res", (1920, 1080)),
                        camera_name=str(d.get("camera_name", "Unbekannt")),
                        room_objects=d.get("room_objects", [])
                    )

                    if rendered_img is not None:
                        core_id: int = get_current_cpu_core()
                        self.frame_ready.emit(rendered_img, core_id)

                except Exception as e:
                    print(f"Render Thread Error: {e}")

            self.msleep(5)

    def stop(self) -> None:
        self.is_running = False
        self.wait()


class Server3DView(QOpenGLWidget):
    def __init__(self, title: str = "Kamera", is_master: bool = False) -> None:
        super().__init__()
        self.title: str = title
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

        self.rotation_yaw: float = -30.0
        self.rotation_pitch: float = -20.0
        self.zoom_level: float = 1.0
        self.last_mouse_pos: QPoint = QPoint()
        self.mouse_sensitivity: float = 0.5

        self.last_scene_data: Optional[dict[str, Any]] = None
        self._current_qimage: Optional[QImage] = None
        self.is_master: bool = is_master
        self._needs_render: bool = False

        self.last_frame_time: float = time.time()
        self.last_process_time: float = time.process_time()
        self.current_fps: float = 0.0
        self.current_cores_used: float = 0.0
        self.last_core_id: int = -1
        self.total_cores: Union[int, str] = os.cpu_count() or "?"

        self.renderer: CalibrationRenderer = CalibrationRenderer()
        self.render_worker: RenderWorker = RenderWorker(self.renderer)
        self.render_worker.frame_ready.connect(self._on_frame_ready)
        self.render_worker.start()

        refresh_rate: int
        if self.is_master:
            self.render_worker.setPriority(QThread.Priority.HighPriority)
            refresh_rate = 33
        else:
            self.render_worker.setPriority(QThread.Priority.LowPriority)
            refresh_rate = 50

        self.render_timer: QTimer = QTimer(self)
        self.render_timer.timeout.connect(self._process_render_queue)
        self.render_timer.start(refresh_rate)

    def update_scene(self, person_data_list: list[dict[str, Any]], room_dims: dict[str, float], camera_points_3d: list[np.ndarray], pixel_points: Optional[list[Any]] = None, camera_pos_label: Optional[str] = None,
                     custom_rectangles: Optional[list[dict[str, Any]]] = None, img_size: tuple[int, int] = (1920, 1080), camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None,
                     all_cameras_3d: Optional[dict[str, Any]] = None, active_ray_cameras: Optional[dict[str, Any]] = None, render_graphics: bool = True, custom_res: tuple[int, int] = (1920, 1080),
                     room_objects: Optional[list[dict[str, Any]]] = None) -> None:
        if person_data_list:
            for p in person_data_list:
                if 'pos' not in p and 'x' in p:
                    p['pos'] = np.array([p['x'], p['y'], p['z']], dtype=np.float32)
                elif 'pos' in p and not isinstance(p['pos'], np.ndarray):
                    p['pos'] = np.array(p['pos'], dtype=np.float32)

        self.last_scene_data = {
            "camera_name": self.title,
            "persons": person_data_list,
            "room": room_dims,
            "points_3d": camera_points_3d,
            "points_2d": pixel_points or [],
            "cam_pos": camera_pos_label,
            "custom_rects": custom_rectangles or [],
            "img_size": img_size,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "all_cameras_3d": all_cameras_3d or {},
            "active_ray_cameras": active_ray_cameras or {},
            "render_graphics": render_graphics,
            "custom_res": custom_res,
            "room_objects": room_objects or []
        }
        self._needs_render = True

    def _process_render_queue(self) -> None:
        if self._needs_render and not self.render_worker.has_new_data:
            if not self.last_scene_data:
                return
            d: dict[str, Any] = dict(self.last_scene_data)
            d["rotation_yaw"] = self.rotation_yaw
            d["rotation_pitch"] = self.rotation_pitch
            d["zoom_level"] = self.zoom_level
            self.render_worker.current_data = d
            self.render_worker.has_new_data = True
            self._needs_render = False

    def _on_frame_ready(self, rendered_img: np.ndarray, core_id: int) -> None:
        now: float = time.time()
        now_process: float = time.process_time()
        dt_real: float = now - self.last_frame_time
        dt_process: float = now_process - self.last_process_time
        self.last_frame_time, self.last_process_time = now, now_process

        if dt_real > 0:
            self.current_fps = (self.current_fps * 0.9) + ((1.0 / dt_real) * 0.1)
            self.current_cores_used = (self.current_cores_used * 0.9) + ((dt_process / dt_real) * 0.1)

        self.last_core_id = core_id

        h: int
        w: int
        ch: int
        h, w, ch = rendered_img.shape
        self._current_qimage = QImage(rendered_img.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
        self.update()

    def paintGL(self) -> None:
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(17, 17, 17))

        if self._current_qimage and not self._current_qimage.isNull():
            target_rect: Any = self.rect()
            scaled_img_rect: Any = self._current_qimage.rect()

            factor: float = min(target_rect.width() / scaled_img_rect.width(), target_rect.height() / scaled_img_rect.height())
            new_w: int = int(scaled_img_rect.width() * factor)
            new_h: int = int(scaled_img_rect.height() * factor)
            x: int = (target_rect.width() - new_w) // 2
            y: int = (target_rect.height() - new_h) // 2

            painter.drawImage(x, y, self._current_qimage.scaled(new_w, new_h, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                                Qt.TransformationMode.FastTransformation))
        else:
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "🔴 PERFORMANCE MODE AKTIV (Grafik pausiert)")

        self._draw_hud(painter)
        painter.end()

    def _draw_hud(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 255, 150))
        font: Any = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(10, 25, f"FPS: {int(self.current_fps)}")

        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor(150, 200, 255))
        core_text: str = f"CPU Core: {self.last_core_id} / {self.total_cores}" if self.last_core_id >= 0 else "CPU Core: ?"
        painter.drawText(10, 45, core_text)

        painter.setPen(QColor(255, 150, 200))
        painter.drawText(10, 65, f"App Load: {self.current_cores_used:.1f} Cores")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            dx: int = event.pos().x() - self.last_mouse_pos.x()
            dy: int = event.pos().y() - self.last_mouse_pos.y()
            self.rotation_yaw += dx * self.mouse_sensitivity
            self.rotation_pitch = max(-89.0, min(89.0, self.rotation_pitch - (dy * self.mouse_sensitivity)))
            self.last_mouse_pos = event.pos()
            self._needs_render = True

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.zoom_level = max(0.1, min(5.0, self.zoom_level + ((event.angleDelta().y() / 120.0) * 0.1)))
        self._needs_render = True