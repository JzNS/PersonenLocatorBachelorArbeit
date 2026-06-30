import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from server.core.logger import get_logger
from server.core.exceptions import RenderingError

logger = get_logger("server.rendering")
from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.gui.logic.label.render.CalibrationRenderer import CalibrationRenderer, RenderColors


class Renderer3D:
    """Kümmert sich exklusiv um das Zeichnen der virtuellen 3D-Welt."""

    @staticmethod
    def render_3d_scene(world_points_3d: list[np.ndarray], rotation_yaw: float, rotation_pitch: float, mesh_connections: list[Any], angles_to_measure: list[Any],
                        point_labels: list[Any], person_results: Optional[list[dict[str, Any]]] = None, room_dims: Optional[dict[str, float]] = None, view_options: Optional[dict[str, bool]] = None,
                        zoom_level: float = 1.0, pixel_points: Optional[list[Any]] = None, camera_pos_label: Optional[str] = None, custom_rectangles: Optional[list[dict[str, Any]]] = None) -> np.ndarray:

        view_opts: dict[str, bool] = view_options or {}

        cam_pos_world: Optional[np.ndarray] = None
        if pixel_points and world_points_3d and len(pixel_points) >= 4:
            pose: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = GeometryMath.get_camera_pose(pixel_points, world_points_3d, custom_rectangles)
            if pose:
                rvec, tvec, _, _ = pose
                R, _ = cv2.Rodrigues(rvec)
                cam_pos_world = -np.dot(R.T, tvec).flatten()

        canvas: Optional[np.ndarray]
        view_ctx: Optional[dict[str, Any]]
        canvas, view_ctx = Renderer3D._setup_3d_canvas(world_points_3d, room_dims, rotation_yaw, rotation_pitch,
                                                       cam_pos_world)
        if canvas is None or view_ctx is None:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        cam_pt_2d: Optional[tuple[int, int]] = None
        if cam_pos_world is not None:
            cam_pt_2d = GeometryMath.project_3d_to_2d(cam_pos_world, view_ctx['center'], view_ctx['rot_params'],
                                                      view_ctx['offset'], zoom_level)
            if view_opts.get("show_camera_world", True) and cam_pt_2d is not None:
                Renderer3D._draw_3d_camera_icon(canvas, cam_pt_2d)
            if hasattr(Renderer3D, '_draw_calculated_camera_pose'):
                getattr(Renderer3D, '_draw_calculated_camera_pose')(canvas, pixel_points, world_points_3d, custom_rectangles, view_ctx,
                                                        zoom_level)

        if room_dims:
            if view_opts.get("show_real_world", True):
                Renderer3D._draw_3d_room_frame(canvas, room_dims, view_ctx, zoom_level)
            if view_opts.get("show_floor_grid", True):
                Renderer3D._draw_3d_floor_grid(canvas, room_dims, view_ctx, zoom_level)

        if world_points_3d and view_opts.get("show_camera_world", True):
            Renderer3D._draw_3d_calibration_mesh(canvas, world_points_3d, mesh_connections, point_labels, view_ctx,
                                                 zoom_level)

        if custom_rectangles and view_opts.get("show_real_world", True):
            Renderer3D._draw_3d_custom_rectangles(canvas, custom_rectangles, view_ctx, zoom_level)

        if pixel_points and world_points_3d:
            show_rays: bool = view_opts.get("show_rays", True)
            Renderer3D._draw_3d_ransac_analysis(canvas, pixel_points, world_points_3d, custom_rectangles, view_ctx,
                                                zoom_level, show_rays=show_rays)

        if person_results:
            active_cam_pos: Optional[tuple[int, int]] = cam_pt_2d if view_opts.get("show_sightlines", True) else None
            Renderer3D._draw_3d_persons(canvas, person_results, view_ctx, zoom_level, pixel_points, world_points_3d,
                                        view_opts, active_cam_pos, custom_rectangles)

        return canvas

    @staticmethod
    def _setup_3d_canvas(world_points_3d: list[np.ndarray], room_dims: Optional[dict[str, float]], yaw: float, pitch: float, cam_pos_world: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
        """Initialisiert das Canvas und berechnet den Mittelpunkt der Szene für die Projektion."""
        try:
            W_IMG: int = 1920
            H_IMG: int = 1080
            canvas: np.ndarray = np.zeros((H_IMG, W_IMG, 3), dtype=np.uint8)

            nodes: list[np.ndarray] = []
            if world_points_3d:
                nodes.extend(world_points_3d)

            if room_dims:
                W: float = float(room_dims.get("width", 600.0))
                H: float = float(room_dims.get("height", 250.0))
                D: float = float(room_dims.get("depth", 800.0))
                nodes.extend([np.array([0, 0, 0]), np.array([W, 0, 0]), np.array([0, 0, D]), np.array([W, 0, D]), np.array([0, H, 0]), np.array([W, H, D])])

            if cam_pos_world is not None:
                nodes.append(cam_pos_world)

            if len(nodes) > 0:
                center: np.ndarray = np.mean(nodes, axis=0)
            else:
                return None, None

            return canvas, {'center': center, 'rot_params': {'yaw': float(np.radians(yaw)), 'pitch': float(np.radians(pitch))},
                            'offset': (W_IMG // 2, H_IMG // 2)}
        except Exception as e:
            err = RenderingError("Renderer3D._setup_3d_canvas", str(e))
            logger.error(str(err))
            return None, None

    @staticmethod
    def _draw_3d_persons(canvas: np.ndarray, persons: list[dict[str, Any]], ctx: dict[str, Any], zoom: float, pixel_points: Any, world_points: Any, view_options: dict[str, bool], cam_pos_2d: Optional[tuple[int, int]],
                         custom_rectangles: Optional[list[dict[str, Any]]] = None) -> None:
        """Zeichnet die Personen als 3D-Säulen oder Skelett, abhängig von den verfügbaren Daten und Einstellungen."""
        show_skel: bool = view_options.get("show_skeleton_3d", True)

        for p in persons:
            base_3d: np.ndarray = np.array(p['pos'], dtype=np.float32)
            color: tuple[int, int, int] = RenderColors.PERSON_STAND
            skel_3d: dict[int, np.ndarray] = p.get('skeleton_3d', {})

            if not skel_3d and show_skel and pixel_points and world_points and len(pixel_points) >= 4 and len(world_points) >= 4:
                skel_3d = GeometryMath.lift_skeleton_to_3d(p.get('keypoints', []), base_3d, pixel_points, world_points,
                                                           custom_rectangles)

            if skel_3d:
                label_pos: Optional[tuple[int, int]] = Renderer3D._draw_person_skeleton(canvas, skel_3d, color, ctx, zoom, cam_pos_2d)
            else:
                label_pos = Renderer3D._draw_person_pillar(canvas, base_3d, float(p.get('height', 180.0)), color, ctx, zoom,
                                                           cam_pos_2d)

            name_label: str = str(p.get('status', f"ID:{p['id']}"))
            if label_pos is not None:
                CalibrationRenderer.draw_hud_text(canvas, f"{name_label} | {float(p.get('height', 0)):.0f}cm",
                                                  (label_pos[0] + 15, label_pos[1] - 15), color, 0.8, 1)

    @staticmethod
    def _draw_person_skeleton(canvas: np.ndarray, skel_3d: dict[int, np.ndarray], color: tuple[int, int, int], ctx: dict[str, Any], zoom: float, cam_pos_2d: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """Zeichnet das 3D-Skelett der Person. Zusätzlich wird eine vertikale Linie zum Boden gezeichnet, um die genaue Position zu verdeutlichen."""
        projected_joints: dict[int, tuple[int, int]] = {
            j_id: GeometryMath.project_3d_to_2d(pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for
            j_id, pos in skel_3d.items()}

        if hasattr(Renderer3D, '_draw_sightlines'):
            getattr(Renderer3D, '_draw_sightlines')(canvas, projected_joints.values(), cam_pos_2d)

        for pt_2d in projected_joints.values():
            cv2.circle(canvas, pt_2d, 4, color, -1, cv2.LINE_AA)

        for s, e in CalibrationRenderer.SKELETON_LINKS:
            if s in projected_joints and e in projected_joints:
                cv2.line(canvas, projected_joints[s], projected_joints[e], RenderColors.GRAY_LIGHT, 2, cv2.LINE_AA)

        if 11 in skel_3d and 12 in skel_3d:
            hip_center: np.ndarray = (skel_3d[11] + skel_3d[12]) / 2.0
            p_hip: tuple[int, int] = GeometryMath.project_3d_to_2d(hip_center, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            ground_pos: np.ndarray = np.array([hip_center[0], 0.0, hip_center[2]], dtype=np.float32)
            p_foot: tuple[int, int] = GeometryMath.project_3d_to_2d(ground_pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            CalibrationRenderer.draw_dashed_line(canvas, p_hip, p_foot, RenderColors.GRAY_DARK, 2, 6)

            cv2.ellipse(canvas, p_foot, (20, 8), 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, p_foot, 2, RenderColors.WHITE, -1, cv2.LINE_AA)

        head_id: int = 0 if 0 in skel_3d else (5 if 5 in skel_3d else -1)
        if head_id != -1 and head_id in projected_joints:
            return projected_joints[head_id]
        return list(projected_joints.values())[0] if projected_joints else None

    @staticmethod
    def _draw_3d_camera_icon(canvas: np.ndarray, pt: tuple[int, int]) -> None: ...
    @staticmethod
    def _draw_3d_room_frame(canvas: np.ndarray, dims: dict[str, float], ctx: dict[str, Any], zoom: float) -> None: ...
    @staticmethod
    def _draw_3d_floor_grid(canvas: np.ndarray, dims: dict[str, float], ctx: dict[str, Any], zoom: float) -> None: ...
    @staticmethod
    def _draw_3d_calibration_mesh(canvas: np.ndarray, world_points: list[np.ndarray], connections: list[Any], labels: list[Any], ctx: dict[str, Any], zoom: float) -> None: ...
    @staticmethod
    def _draw_3d_custom_rectangles(canvas: np.ndarray, rects: list[dict[str, Any]], ctx: dict[str, Any], zoom: float) -> None: ...
    @staticmethod
    def _draw_3d_ransac_analysis(canvas: np.ndarray, pixel_pts: list[Any], world_pts: list[np.ndarray], rects: Optional[list[dict[str, Any]]], ctx: dict[str, Any], zoom: float, show_rays: bool) -> None: ...
    @staticmethod
    def _draw_person_pillar(canvas: np.ndarray, pos: np.ndarray, height: float, color: tuple[int, int, int], ctx: dict[str, Any], zoom: float, cam_pos: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]: ...
