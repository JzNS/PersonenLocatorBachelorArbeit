import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from PyQt6.QtCore import pyqtSignal, QObject

from client.gui.logic.PersonManager import PersonManager
from client.gui.logic.label.math.CalibrationMath import CalibrationMath
from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.gui.logic.label.render.Renderer2D import Renderer2D
from client.gui.logic.label.render.Renderer3D import Renderer3D
from client.gui.logic.label.utilityClasses.utilityClassCalibration.Tooblox.ToolboxConfigLoader import \
    ToolboxConfigLoader
from client.gui.logic.label.utilityClasses.utilityClassCalibration.Tooblox.ZoneManager import ZoneManager
from client.utils.ConfigManager import ConfigManager


class CalibrationToolbox(QObject):
    """Zentraler Status-Container. Delegiert alle Aufgaben an spezialisierte Manager."""
    zones_changed: pyqtSignal = pyqtSignal()

    POINT_LABELS: list[str] = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]
    MESH_CONNECTIONS: list[tuple[int, int]] = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (6, 7), (7, 5), (5, 4), (0, 4), (1, 6), (2, 5), (3, 7)]
    ANGLES_TO_MEASURE: list[tuple[int, int, int]] = [(0, 1, 2), (0, 1, 4), (0, 2, 4), (1, 0, 3), (1, 0, 6), (1, 3, 6), (2, 0, 3), (2, 0, 5),
                         (2, 3, 5), (3, 1, 2), (3, 1, 7), (3, 2, 7), (4, 0, 6), (4, 0, 5), (4, 6, 5), (5, 4, 7),
                         (5, 4, 2), (5, 7, 2), (6, 4, 7), (6, 4, 1), (6, 7, 1), (7, 6, 5), (7, 6, 3), (7, 5, 3)]

    def __init__(self, client_name: str) -> None:
        super().__init__()
        self.client_name: str = client_name

        self.person_manager: PersonManager = PersonManager()
        self.zone_manager: ZoneManager = ZoneManager(client_name)

        self.zone_manager.zones_changed.connect(self.zones_changed.emit)

        self.pixel_points: list[tuple[int, int]] = []
        self.world_points_3d: list[np.ndarray] = []
        self.custom_rectangles: list[dict[str, Any]] = []

        self.rotation_yaw: float = -30.0
        self.rotation_pitch: float = -20.0
        self.mouse_sensitivity: float = 0.5
        self.zoom_level: float = 1.0

        self.real_room_dims: dict[str, float] = {}
        self.view_options: dict[str, Any] = {}
        self.current_resolution: tuple[int, int] = (1920, 1080)

        self._load_config()

    def _load_config(self) -> None:
        """Bezieht den Zustand fertig formatiert über den ConfigLoader."""
        state: dict[str, Any] = ToolboxConfigLoader.load_full_state(self.client_name)

        self.view_options = state["view_options"]
        self.real_room_dims = state["real_room_dims"]
        self.custom_rectangles = state["custom_rectangles"]

        self.zone_manager.load_zones(state["dead_zones"], state["mirror_zones"])
        cam_conf: dict[str, Any] = ConfigManager.load_camera_config().get(self.client_name, {})
        res: list[int] = cam_conf.get("resolution", [1920, 1080])
        self.current_resolution = (int(res[0]), int(res[1]))
        if state["points_3d_data"]:
            self.world_points_3d = [np.array([p["x"], p["y"], p["z"]], dtype=np.float32) for p in
                                    state["points_3d_data"]]

            self.pixel_points = []
            for p in state["points_3d_data"]:
                px: Optional[int] = p.get("px")
                py: Optional[int] = p.get("py")
                safe_px: int = int(px) if px is not None else 0
                safe_py: int = int(py) if py is not None else 0
                self.pixel_points.append((safe_px, safe_py))

        logging.info(f"Toolbox für {self.client_name} geladen.")
        self.zones_changed.emit()

    def draw_toolbox_ui(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Delegiert das Zeichnen aller UI-Elemente an den Renderer2D. Alle notwendigen Daten werden übergeben."""
        if frame is None:
            return frame
        cur_dead: list[Any]
        cur_mirror: list[Any]
        cur_dead, cur_mirror = self.zone_manager.get_zones()

        return Renderer2D.render_2d_toolbox(
            frame=frame, scale=scale,
            dead_zones=cur_dead, mirror_zones=cur_mirror,
            pixel_points=self.pixel_points, custom_rectangles=self.custom_rectangles,
            mesh_connections=self.MESH_CONNECTIONS, world_points=self.world_points_3d,
            room_dims=self.real_room_dims, view_options=self.view_options
        )

    def generate_3d_preview(self, person_results: Optional[list[dict[str, Any]]]) -> np.ndarray:
        """Delegiert das Zeichnen der 3D-Vorschau an Renderer3D. Alle notwendigen Daten werden übergeben."""
        return Renderer3D.render_3d_scene(
            world_points_3d=self.world_points_3d,
            rotation_yaw=self.rotation_yaw,
            rotation_pitch=self.rotation_pitch,
            mesh_connections=self.MESH_CONNECTIONS,
            angles_to_measure=self.ANGLES_TO_MEASURE,
            point_labels=self.POINT_LABELS,
            person_results=person_results,
            room_dims=self.real_room_dims,
            view_options=self.view_options,
            zoom_level=self.zoom_level,
            pixel_points=self.pixel_points,
            camera_pos_label=None,
            custom_rectangles=self.custom_rectangles,
            img_size=self.current_resolution
        )

    def analyze_body_metrics(self, keypoints: list[dict[str, Any]], pos_3d: np.ndarray, frame: Optional[np.ndarray] = None,
                             correction_factor: float = 1.0) -> dict[str, Any]:
        """Delegiert die Analyse der Körpermetriken an CalibrationMath."""
        return CalibrationMath.analyze_body_metrics(
            keypoints, pos_3d, self.pixel_points, self.world_points_3d,
            correction_factor, frame, self.custom_rectangles,
            img_size=self.current_resolution
        )

    def calculate_height_and_confidence(self, pixel_bbox: list[int], pos_3d: np.ndarray, keypoints: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Delegiert die Berechnung von Körpergröße und Entfernung an CalibrationMath."""
        final_h: float
        used_skel: bool
        final_h, used_skel = CalibrationMath.calculate_person_height(
            pixel_bbox, pos_3d, keypoints, self.pixel_points,
            self.world_points_3d, self.custom_rectangles,
            img_size=self.current_resolution
        )
        dist: float
        conf: float
        dist, conf = CalibrationMath.calculate_confidence_and_distance(pos_3d)
        return {"height": round(final_h, 1), "distance": dist, "confidence": conf, "is_skeleton": used_skel}

    def project_2d_to_3d(self, px: int, py: int) -> Optional[np.ndarray]:
        """Delegiert die Projektion von 2D in 3D an GeometryMath."""
        return GeometryMath.project_2d_to_3d(
            px, py, self.pixel_points, self.world_points_3d,
            self.custom_rectangles,
            img_size=self.current_resolution
        )

    def get_geometric_analysis(self) -> dict[str, Any]:
        """Delegiert die Berechnung von Distanzen, Winkeln und Raumverhältnissen an CalibrationMath. Alle notwendigen Daten werden übergeben."""
        return CalibrationMath.generate_geometric_analysis(self.world_points_3d, self.POINT_LABELS,
                                                           self.MESH_CONNECTIONS, self.ANGLES_TO_MEASURE)

    def update_zoom(self, delta_steps: int) -> None:
        """Aktualisiert den Zoom-Level basierend auf der Anzahl der Mausrad-Schritte. Nutzt np.clip, um den Zoom auf einen Bereich von 0.1 bis 5.0 zu begrenzen."""
        self.zoom_level = float(np.clip(self.zoom_level + (delta_steps * 0.1), 0.1, 5.0))

    def update_rotation(self, dx: int, dy: int) -> None:
        """Aktualisiert die Rotationswinkel basierend auf den Mausbewegungen. Nutzt np.clip, um die Pitch-Rotation auf einen Bereich von -89 bis 89 Grad zu begrenzen."""
        self.rotation_yaw += dx * self.mouse_sensitivity
        self.rotation_pitch = float(np.clip(self.rotation_pitch - dy * self.mouse_sensitivity, -89, 89))

    def update_room_dimensions(self, w: Union[float, str], h: Union[float, str], d: Union[float, str]) -> None:
        """Aktualisiert die realen Raumdimensionen, die für die 3D-Projektion und Analyse verwendet werden. Alle Werte werden in Float umgewandelt, um Konsistenz zu gewährleisten."""
        self.real_room_dims = {"width": float(w), "height": float(h), "depth": float(d)}