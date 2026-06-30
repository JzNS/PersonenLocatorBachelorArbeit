import cv2
import json
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Callable, Set

from server.core.math.GeometryMath import GeometryMath
from server.rendering.DrawingUtils import DrawingUtils

_CAM_COLORS: dict[str, tuple[int, int, int]] = {
    "CAMERA_1": (80,  80,  255),
    "CAMERA_2": (255, 150, 80),
    "CAMERA_3": (80,  255, 80),
    "CAMERA_4": (80,  255, 255),
}
_SKELETON_LINKS: list[tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12),
]


class SceneRenderer:
    """
    Rendert die 3D-Szene mit Raum, Kameras, Rohdaten-Skeletten und fusionierten
    Skeletten auf einen OpenCV-Canvas.

    Hat keine eigenen Tracking-Zustände – nimmt alle nötigen Daten als Parameter.
    """

    def render_3d_scene(
        self,
        world_points_3d: list[np.ndarray],
        rotation_yaw: float,
        rotation_pitch: float,
        mesh_connections: list[Any],
        angles_to_measure: list[Any],
        point_labels: list[Any],
        person_results: Optional[list[dict[str, Any]]] = None,
        room_dims: Optional[dict[str, float]] = None,
        view_options: Optional[dict[str, bool]] = None,
        zoom_level: float = 1.0,
        pixel_points: Optional[list[Any]] = None,
        camera_pos_label: Optional[str] = None,
        custom_rectangles: Optional[list[dict[str, Any]]] = None,
        img_size: tuple[int, int] = (1920, 1080),
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        all_cameras_3d: Optional[dict[str, Any]] = None,
        active_ray_cameras: Optional[dict[str, Any]] = None,
        render_graphics: bool = True,
        custom_res: tuple[int, int] = (1920, 1080),
        camera_name: str = "Unbekannt",
        room_objects: Optional[list[dict[str, Any]]] = None,
        on_master_fusion: Optional[Callable[[Optional[np.ndarray], list[dict[str, Any]], dict[str, Any], float, bool, dict[str, Any], Optional[list[dict[str, Any]]], Optional[dict[str, float]]], None]] = None,
    ) -> Optional[np.ndarray]:
        """
        Erstellt das vollständige 3D-Szenenbild.

        `on_master_fusion` wird im Master-View aufgerufen mit (canvas, persons, ctx, zoom, render_graphics, fusion_settings, room_objects, room_dims)
        und soll die Fusions-Skelette rendern / berechnen.
        """
        is_master: bool = bool(all_cameras_3d)
        width_img, height_img = custom_res

        canvas: Optional[np.ndarray] = np.zeros((height_img, width_img, 3), dtype=np.uint8) if render_graphics else None
        zoom: float = zoom_level * (width_img / 1920.0)
        if not is_master:
            zoom *= 0.6

        view_opts: dict[str, bool] = view_options or {"show_real_world": True, "show_camera_world": True}

        cam_pos_world: Optional[np.ndarray]
        R_inv: Optional[np.ndarray]
        K_mat: Optional[np.ndarray]
        dist_c: Optional[np.ndarray]
        cam_pos_world, R_inv, K_mat, dist_c = self._compute_camera_pose(
            pixel_points, world_points_3d, custom_rectangles, img_size, camera_matrix, dist_coeffs, camera_name
        )

        ctx: dict[str, Any] = self._build_view_context(room_dims, rotation_yaw, rotation_pitch, width_img, height_img)

        all_cams_2d: dict[str, tuple[int, int]] = self._project_cameras(
            is_master, all_cameras_3d, cam_pos_world, person_results, ctx, zoom
        )

        if render_graphics and canvas is not None:
            if room_dims and view_opts.get("show_real_world", True):
                DrawingUtils.draw_room_frame(canvas, room_dims, ctx, zoom)
            if custom_rectangles and view_opts.get("show_real_world", True):
                DrawingUtils.draw_custom_rectangles(canvas, custom_rectangles, ctx, zoom)
            if room_objects and view_opts.get("show_real_world", True):
                DrawingUtils.draw_room_objects(canvas, room_objects, ctx, zoom)

        if person_results:
            effective_cameras: Optional[dict[str, Any]]
            active_cameras: Optional[dict[str, Any]]
            effective_cameras, active_cameras = self._resolve_cameras(
                is_master, cam_pos_world, R_inv, K_mat, dist_c,
                person_results, all_cameras_3d, active_ray_cameras
            )
            self.draw_persons_3d(
                canvas, person_results, ctx, zoom,
                pixel_points, world_points_3d, view_opts,
                all_cams_2d, effective_cameras, active_cameras,
                is_master=is_master, render_graphics=render_graphics,
                on_master_fusion=on_master_fusion,
                room_objects=room_objects,
                room_dims=room_dims,
            )

        if render_graphics and canvas is not None and view_opts.get("show_camera_world", True):
            DrawingUtils.draw_cameras(canvas, all_cams_2d)

        return canvas

    def draw_persons_3d(
        self,
        canvas: Optional[np.ndarray],
        persons: list[dict[str, Any]],
        ctx: dict[str, Any],
        zoom: float,
        pixel_points: Any,
        world_points: Any,
        view_options: Optional[dict[str, Any]] = None,
        all_cams_2d: Optional[dict[str, tuple[int, int]]] = None,
        all_cameras_3d: Optional[dict[str, Any]] = None,
        active_ray_cameras: Optional[dict[str, Any]] = None,
        is_master: bool = False,
        render_graphics: bool = True,
        on_master_fusion: Optional[Callable[[Optional[np.ndarray], list[dict[str, Any]], dict[str, Any], float, bool, dict[str, Any], Optional[list[dict[str, Any]]], Optional[dict[str, float]]], None]] = None,
        room_objects: Optional[list[dict[str, Any]]] = None,
        room_dims: Optional[dict[str, float]] = None,
    ) -> None:
        """Rendert Rohdaten-Strahlen, 3D-Skelette und optionale Fusion."""
        active_ray_cams: dict[str, Any] = active_ray_cameras or {}
        all_cams_3d: dict[str, Any]     = all_cameras_3d or {}
        cams_2d: dict[str, tuple[int, int]]        = all_cams_2d or {}

        def to_pt(pt: tuple[float, float]) -> tuple[int, int]:
            return (int(round(pt[0])), int(round(pt[1])))

        fusion_settings: dict[str, Any]  = self._parse_dict_or_bool(
            active_ray_cams.get("MASTER_FUSION", {}),
            default={"master": True, "joints": {}, "show_rays": False, "show_points": True, "show_bones": True},
        )
        orange_color: tuple[int, int, int] = (0, 165, 255)

        for p in persons:
            p["calculated_rays"] = {}
            origin_cam: str   = str(p.get("cam_name", "UNKNOWN_CAM"))
            cam_data: Any     = all_cams_3d.get(origin_cam)
            cam_settings: dict[str, Any] = self._parse_dict_or_bool(
                active_ray_cams.get(origin_cam, {}),
                default={"master": True, "show_rays": True, "show_single_rays": True,
                         "show_points": True, "show_bones": True},
            )

            show_rays: bool   = bool(cam_settings.get("show_rays" if is_master else "show_single_rays", True))
            show_points: bool = bool(cam_settings.get("show_points", True))
            show_bones: bool  = bool(cam_settings.get("show_bones",  True))

            default_color: tuple[int, int, int] = _CAM_COLORS.get(origin_cam, (200, 255, 255))

            skel_3d: dict[int, np.ndarray]          = self._parse_skeleton_3d(p)
            projected_joints: dict[int, tuple[int, int]] = self._project_skeleton(skel_3d, ctx, zoom)
            all_joint_ids: Set[int]    = set([int(kp.get("id")) for kp in p.get("keypoints", []) if kp.get("id") is not None] + list(skel_3d.keys()))

            target_z: Optional[float] = self._estimate_target_z(is_master, cam_data, p, skel_3d)

            self._process_raycasting(
                p, all_joint_ids, cam_data, cam_settings, origin_cam,
                skel_3d, projected_joints, target_z,
                ctx, zoom, canvas, render_graphics,
                show_rays, is_master, cams_2d, orange_color, default_color, to_pt,
            )

            if render_graphics and canvas is not None:
                self._draw_raw_skeleton(canvas, p, projected_joints, skel_3d, ctx, zoom,
                                        show_points, show_bones, default_color, to_pt)

        if is_master and fusion_settings.get("master", True) and on_master_fusion:
            on_master_fusion(canvas, persons, ctx, zoom, render_graphics, fusion_settings, room_objects, room_dims)

    @staticmethod
    def _compute_camera_pose(pixel_pts: Optional[list[Any]], world_pts: list[np.ndarray], custom_rects: Optional[list[dict[str, Any]]], img_size: tuple[int, int], cam_matrix: Optional[np.ndarray], dist_coeffs: Optional[np.ndarray], cam_name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not pixel_pts or not world_pts or len(pixel_pts) < 4:
            return None, None, None, None
        pose: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = GeometryMath.get_camera_pose(
            pixel_pts, world_pts, custom_rects, img_size=img_size,
            camera_matrix_override=cam_matrix, dist_coeffs_override=dist_coeffs,
            camera_name=cam_name,
        )
        if not pose:
            return None, None, None, None
        rvec, tvec, K_mat, dist_c = pose
        R, _ = cv2.Rodrigues(rvec)
        cam_pos: np.ndarray = (-R.T @ tvec).flatten()
        return cam_pos, R.T, K_mat, dist_c

    @staticmethod
    def _build_view_context(room_dims: Optional[dict[str, float]], yaw: float, pitch: float, width_img: int, height_img: int) -> dict[str, Any]:
        center: np.ndarray
        if room_dims:
            w: float = float(room_dims.get("width",  600.0))
            h: float = float(room_dims.get("height", 250.0))
            d: float = float(room_dims.get("depth",  800.0))
            center = np.array([w / 2.0, h / 2.0, d / 2.0], dtype=np.float32)
        else:
            center = np.array([300.0, 125.0, 400.0], dtype=np.float32)
        return {
            "center":     center,
            "rot_params": {"yaw": float(np.radians(yaw)), "pitch": float(np.radians(pitch))},
            "offset":     (width_img // 2, height_img // 2),
        }

    @staticmethod
    def _project_cameras(is_master: bool, all_cameras_3d: Optional[dict[str, Any]], cam_pos_world: Optional[np.ndarray], person_results: Optional[list[dict[str, Any]]], ctx: dict[str, Any], zoom: float) -> dict[str, tuple[int, int]]:
        all_cams_2d: dict[str, tuple[int, int]] = {}
        if is_master and all_cameras_3d:
            for name, data in all_cameras_3d.items():
                all_cams_2d[name] = GeometryMath.project_3d_to_2d(
                    data["pos"], ctx["center"], ctx["rot_params"], ctx["offset"], zoom
                )
        elif cam_pos_world is not None:
            c_name: str = (person_results[0].get("cam_name", "SINGLE_CAM") if person_results else "SINGLE_CAM")
            all_cams_2d[c_name] = GeometryMath.project_3d_to_2d(
                cam_pos_world, ctx["center"], ctx["rot_params"], ctx["offset"], zoom
            )
        return all_cams_2d

    @staticmethod
    def _resolve_cameras(is_master: bool, cam_pos_world: Optional[np.ndarray], R_inv: Optional[np.ndarray], K_mat: Optional[np.ndarray], dist_c: Optional[np.ndarray],
                         person_results: list[dict[str, Any]], all_cameras_3d: Optional[dict[str, Any]], active_ray_cameras: Optional[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalisiert Kamera-Objekte für Master- und Einzelkamera-Ansicht."""
        if is_master:
            return all_cameras_3d or {}, active_ray_cameras or {}

        real_cam: str = str(person_results[0].get("cam_name", "SINGLE_CAM") if person_results else "SINGLE_CAM")
        effective: dict[str, Any] = {}
        if cam_pos_world is not None:
            effective[real_cam] = {"pos": cam_pos_world, "R_inv": R_inv, "K": K_mat, "dist": dist_c}

        active: dict[str, Any] = dict(active_ray_cameras) if active_ray_cameras else {}
        if real_cam not in active:
            active[real_cam] = {
                "master": True, "show_rays": True, "show_single_rays": True,
                "show_points": True, "show_bones": True,
                "joints": {i: True for i in range(17)},
            }
        return effective, active

    @staticmethod
    def _parse_skeleton_3d(p: dict[str, Any]) -> dict[int, np.ndarray]:
        """Parst Skelett-Daten aus verschiedenen möglichen Quellen/Formaten."""
        raw: Any = p.get("skeleton_3d") or p.get("metrics", {}).get("skeleton_3d", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return {}

        skel: dict[int, np.ndarray] = {}
        if isinstance(raw, dict):
            for j_id_str, pos in raw.items():
                j_id: int = int(j_id_str)
                if isinstance(pos, dict):
                    skel[j_id] = np.array([float(pos.get("x",0.0)), float(pos.get("y",0.0)), float(pos.get("z",0.0))], dtype=np.float32)
                elif isinstance(pos, (list, np.ndarray)) and len(pos) == 3:
                    skel[j_id] = np.array(pos, dtype=np.float32)
        return skel

    @staticmethod
    def _project_skeleton(skel_3d: dict[int, np.ndarray], ctx: dict[str, Any], zoom: float) -> dict[int, tuple[int, int]]:
        projected: dict[int, tuple[int, int]] = {}
        for j_id, pos in skel_3d.items():
            pt: tuple[int, int] = GeometryMath.project_3d_to_2d(pos, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            if pt:
                projected[j_id] = pt
        return projected

    @staticmethod
    def _estimate_target_z(is_master: bool, cam_data: Optional[dict[str, Any]], p: dict[str, Any], skel_3d: dict[int, np.ndarray]) -> Optional[float]:
        """Schätzt Boden-Z für Einzelkamera-Fallback (nur wenn noch kein Skelett vorhanden)."""
        if is_master or not cam_data or skel_3d:
            return None
        valid_kps: list[dict[str, Any]] = [k for k in p.get("keypoints", []) if float(k.get("c", 0.0)) > 0.4]
        if not valid_kps:
            return None

        lowest_kp: dict[str, Any] = max(valid_kps, key=lambda k: float(k.get("y", 0.0)))
        px: float = float(lowest_kp["x"])
        py: float = float(lowest_kp["y"])
        k_mat: Optional[np.ndarray] = cam_data.get("K")
        dist_c: Optional[np.ndarray] = cam_data.get("dist")
        r_inv: Optional[np.ndarray] = cam_data.get("R_inv")

        if k_mat is None or r_inv is None:
            return None

        pt_raw: np.ndarray = np.array([[[px, py]]], dtype=np.float32)
        if dist_c is not None and np.any(dist_c):
            pt_u: np.ndarray = cv2.undistortPoints(pt_raw, k_mat, dist_c)
            ray_cam: np.ndarray = np.array([pt_u[0][0][0], pt_u[0][0][1], 1.0], dtype=np.float32)
        else:
            ray_cam = np.dot(np.linalg.inv(k_mat), np.array([px, py, 1.0], dtype=np.float32))

        ray_world: np.ndarray = np.dot(r_inv, ray_cam)
        if abs(ray_world[1]) < 1e-6:
            return None
        cam_pos: np.ndarray = cam_data["pos"]
        t_floor: float = -cam_pos[1] / ray_world[1]
        if t_floor <= 0:
            return None
        return float((cam_pos + t_floor * ray_world)[2])

    def _process_raycasting(
        self, p: dict[str, Any], all_joint_ids: Set[int], cam_data: Optional[dict[str, Any]], cam_settings: dict[str, Any], origin_cam: str,
        skel_3d: dict[int, np.ndarray], projected_joints: dict[int, tuple[int, int]], target_z: Optional[float],
        ctx: dict[str, Any], zoom: float, canvas: Optional[np.ndarray], render_graphics: bool,
        show_rays: bool, is_master: bool, all_cams_2d: dict[str, tuple[int, int]], orange_color: tuple[int, int, int], default_color: tuple[int, int, int], to_pt: Callable[[tuple[float, float]], tuple[int, int]],
    ) -> None:
        """Berechnet Kamerastrahlen und zeichnet diese optional auf den Canvas."""
        for j_id in all_joint_ids:
            if not cam_settings.get("joints", {}).get(j_id, True):
                continue

            kp: Optional[dict[str, Any]] = next((k for k in p.get("keypoints", []) if k.get("id") == j_id), None)
            conf: float = float(kp.get("c", 1.0) if kp else 1.0)

            cam_pos_3d: np.ndarray  = np.zeros(3, dtype=np.float32)
            direction: np.ndarray   = np.zeros(3, dtype=np.float32)
            render_color: tuple[int, int, int] = default_color if conf >= 0.80 else orange_color
            cam_pt_2d: Optional[tuple[int, int]]   = all_cams_2d.get(origin_cam)
            P_matrix: Optional[np.ndarray]    = None
            pt_2d_tuple: Optional[tuple[float, float]] = None

            if cam_data and kp:
                cam_pos_3d, direction, P_matrix, pt_2d_tuple = self._compute_ray(cam_data, kp)

                if j_id not in projected_joints and target_z is not None and abs(direction[2]) > 1e-6:
                    t_z: float = (target_z - cam_pos_3d[2]) / direction[2]
                    if t_z > 0:
                        pt_3d: np.ndarray = cam_pos_3d + t_z * direction
                        if -20.0 <= pt_3d[1] <= 260.0:
                            pt_2d: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(pt_3d, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
                            if pt_2d:
                                projected_joints[j_id] = pt_2d
                                skel_3d[j_id] = pt_3d

                if render_graphics and canvas is not None and show_rays and cam_pt_2d and float(np.linalg.norm(direction)) > 1e-6:
                    self._draw_ray(canvas, cam_pos_3d, direction, cam_pt_2d, projected_joints, j_id,
                                   ctx, zoom, render_color, is_master, to_pt)

            j_color: Optional[np.ndarray] = self._extract_joint_color(p, j_id, conf)

            if cam_settings.get("master", True):
                p["calculated_rays"][j_id] = {
                    "cam_pos": cam_pos_3d, "dir": direction,
                    "fallback_pos": skel_3d.get(j_id), "color": j_color,
                    "cam_name": origin_cam, "person_id": p.get("id", 1),
                    "conf": conf, "P_matrix": P_matrix, "pt_2d": pt_2d_tuple,
                }

    @staticmethod
    def _compute_ray(cam_data: dict[str, Any], kp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
        """Berechnet Kameraposition, Richtungsvektor, Projektionsmatrix und 2D-Punkt."""
        cam_pos: np.ndarray = cam_data.get("pos", np.zeros(3, dtype=np.float32))
        r_inv: np.ndarray   = cam_data.get("R_inv", np.eye(3, dtype=np.float32))
        k_mat: np.ndarray   = cam_data.get("K", np.eye(3, dtype=np.float32))
        dist_c: Optional[np.ndarray]  = cam_data.get("dist")

        px: float = float(kp.get("x", 0.0))
        py: float = float(kp.get("y", 0.0))
        pt_2d_tuple: tuple[float, float] = (px, py)

        R: np.ndarray   = r_inv.T
        t: np.ndarray   = -np.dot(R, cam_pos)
        Rt: np.ndarray  = np.hstack((R, t.reshape(3, 1)))
        P: np.ndarray   = np.dot(k_mat, Rt)

        pt_raw: np.ndarray = np.array([[[px, py]]], dtype=np.float32)
        if dist_c is not None and np.any(dist_c):
            pt_u: np.ndarray    = cv2.undistortPoints(pt_raw, k_mat, dist_c)
            ray_cam: np.ndarray = np.array([pt_u[0][0][0], pt_u[0][0][1], 1.0], dtype=np.float32)
        else:
            ray_cam = np.dot(np.linalg.inv(k_mat), np.array([px, py, 1.0], dtype=np.float32))

        ray_world: np.ndarray = np.dot(r_inv, ray_cam)
        norm: float = float(np.linalg.norm(ray_world))
        direction: np.ndarray = ray_world / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)

        return cam_pos, direction, P, pt_2d_tuple

    @staticmethod
    def _draw_ray(canvas: np.ndarray, cam_pos_3d: np.ndarray, direction: np.ndarray, cam_pt_2d: tuple[int, int], projected_joints: dict[int, tuple[int, int]], j_id: int,
                  ctx: dict[str, Any], zoom: float, color: tuple[int, int, int], is_master: bool, to_pt: Callable[[tuple[float, float]], tuple[int, int]]) -> None:
        if is_master:
            ray_end: np.ndarray
            if abs(direction[1]) > 1e-6:
                t_floor: float = -cam_pos_3d[1] / direction[1]
                ray_end = cam_pos_3d + t_floor * direction if t_floor > 0 else cam_pos_3d + direction * 600.0
            else:
                ray_end = cam_pos_3d + direction * 600.0
            p_end: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(ray_end, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            if p_end:
                cv2.line(canvas, to_pt(cam_pt_2d), to_pt(p_end), color, 1, cv2.LINE_AA)
        else:
            if j_id in projected_joints:
                cv2.line(canvas, to_pt(cam_pt_2d), to_pt(projected_joints[j_id]), color, 1, cv2.LINE_AA)
            else:
                ray_end = cam_pos_3d + direction * 400.0
                p_end_s: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(ray_end, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
                if p_end_s:
                    cv2.line(canvas, to_pt(cam_pt_2d), to_pt(p_end_s), color, 1, cv2.LINE_AA)

    @staticmethod
    def _extract_joint_color(p: dict[str, Any], j_id: int, conf: float) -> Optional[np.ndarray]:
        if conf < 0.80:
            return None
        color_hex: Any = (
            p.get("raw_data", {}).get("metrics", {}).get("stable_colors", {}).get(str(j_id))
            or p.get("metrics", {}).get("stable_colors", {}).get(str(j_id))
        )
        if color_hex and isinstance(color_hex, str) and len(color_hex) >= 7:
            try:
                return np.array([int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)], dtype=np.float32)
            except ValueError:
                pass
        return None

    def _draw_raw_skeleton(
        self, canvas: np.ndarray, p: dict[str, Any], projected_joints: dict[int, tuple[int, int]], skel_3d: dict[int, np.ndarray], ctx: dict[str, Any], zoom: float,
        show_points: bool, show_bones: bool, default_color: tuple[int, int, int], to_pt: Callable[[tuple[float, float]], tuple[int, int]],
    ) -> None:
        if show_points:
            for pt in projected_joints.values():
                cv2.circle(canvas, to_pt(pt), 4, default_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, to_pt(pt), 5, (255, 255, 255), 1, cv2.LINE_AA)

        if show_bones:
            for s, e in _SKELETON_LINKS:
                if s in projected_joints and e in projected_joints:
                    cv2.line(canvas, to_pt(projected_joints[s]), to_pt(projected_joints[e]),
                             default_color, 2, cv2.LINE_AA)

        if show_points or show_bones:
            head_id: int = 0 if 0 in projected_joints else (5 if 5 in projected_joints else -1)
            label_pos: Optional[tuple[int, int]]
            if head_id != -1:
                label_pos = projected_joints[head_id]
            else:
                base: np.ndarray = np.array(p.get("pos", np.zeros(3, dtype=np.float32)), dtype=np.float32) + np.array([0.0, 180.0, 0.0], dtype=np.float32)
                label_pos = GeometryMath.project_3d_to_2d(base, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            if label_pos:
                DrawingUtils.draw_hud_text(
                    canvas, f"ID:{p.get('id','?')}",
                    (int(label_pos[0] + 15), int(label_pos[1] - 15)), (255, 255, 0), 0.9, 2,
                )

    def draw_fusion_skeleton(
        self,
        canvas: np.ndarray,
        data: dict[str, Any],
        ctx: dict[str, Any],
        zoom: float,
        to_pt: Callable[[tuple[float, float]], tuple[int, int]],
        show_points: bool = True,
        show_bones: bool = True,
        show_arm_extension: bool = False,
        all_persons: Optional[list[dict[str, Any]]] = None,
        room_objects: Optional[list[dict[str, Any]]] = None,
        room_dims: Optional[dict[str, float]] = None,
    ) -> None:
        """Zeichnet ein einzelnes fusioniertes Skelett inkl. Orientierungspfeil und optionalen Interaktions-Strahlen."""
        skel: dict[int, np.ndarray] = data.get("skel", {})
        if not skel:
            return

        render_color: tuple[int, int, int] = (0, 255, 150) if data.get("is_confirmed") else (0, 100, 200)

        proj: dict[int, tuple[int, int]] = {}
        for j_id, pos in skel.items():
            pt: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(pos, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            if pt:
                proj[j_id] = to_pt(pt)

        if show_points:
            for pt_p in proj.values():
                cv2.circle(canvas, pt_p, 5, render_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt_p, 6, (255, 255, 255), 1, cv2.LINE_AA)

        if show_bones:
            for s, e in _SKELETON_LINKS:
                if s in proj and e in proj:
                    cv2.line(canvas, proj[s], proj[e], render_color, 3, cv2.LINE_AA)

        if show_arm_extension:
            for side, elbow_id, wrist_id in [("left", 7, 9), ("right", 8, 10)]:
                if elbow_id in skel and wrist_id in skel:
                    origin: np.ndarray = skel[wrist_id]
                    direction: np.ndarray = origin - skel[elbow_id]
                    norm: float = float(np.linalg.norm(direction))
                    if norm > 8.0:
                        direction /= norm

                        res: dict[str, Any] = self._find_ray_intersection(
                            origin, direction, room_objects, room_dims, all_persons,
                            exclude_id=data.get("id")
                        )
                        hit_pos: Optional[np.ndarray] = res.get("pos")
                        
                        if hit_pos is not None:
                            p_start: Optional[tuple[int, int]] = proj.get(wrist_id)
                            p_end: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(hit_pos, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
                            
                            if p_start and p_end:
                                overlay = canvas.copy()
                                line_color: tuple[int, int, int] = (255, 255, 255)
                                if res.get("type") == "person":
                                    line_color = (0, 255, 255)
                                elif res.get("type") == "object":
                                    line_color = (0, 165, 255)

                                cv2.line(overlay, p_start, to_pt(p_end), line_color, 1, cv2.LINE_AA)
                                cv2.circle(overlay, to_pt(p_end), 4, line_color, -1, cv2.LINE_AA)

                                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

                        if "pointing" not in data: data["pointing"] = {}
                        data["pointing"][side] = res

        if "forward_vec" in data and all(k in skel for k in [5, 6, 11, 12]):
            com: np.ndarray = (skel[5] + skel[6] + skel[11] + skel[12]) / 4.0
            arrow_end: np.ndarray = com + data["forward_vec"] * 20.0
            pt1: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(com,       ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            pt2: Optional[tuple[int, int]] = GeometryMath.project_3d_to_2d(arrow_end, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            if pt1 and pt2:
                arrow_color: tuple[int, int, int] = (173, 216, 230) if data.get("is_confirmed") else (80, 100, 120)
                cv2.arrowedLine(canvas, to_pt(pt1), to_pt(pt2), arrow_color, 2, cv2.LINE_AA, tipLength=0.4)

        head_pt: Optional[tuple[int, int]] = proj.get(0, proj.get(5))
        if head_pt:
            status: str = f"FUSION ID:{data['id']}" if data.get("is_confirmed") else "Verifiziere..."
            DrawingUtils.draw_hud_text(canvas, status, (head_pt[0] + 15, head_pt[1] - 20), render_color, 0.7, 2)

    def _find_ray_intersection(
        self, origin: np.ndarray, direction: np.ndarray, 
        room_objects: Optional[list[dict[str, Any]]], 
        room_dims: Optional[dict[str, float]] = None,
        persons: Optional[list[dict[str, Any]]] = None,
        exclude_id: Optional[int] = None
    ) -> dict[str, Any]:
        """Findet den nächsten Schnittpunkt des Strahls mit Personen, Objekten oder den Raumwänden."""
        t_min: float = float('inf')
        hit_pos: Optional[np.ndarray] = None
        hit_type: str = "wall"
        hit_label: str = "Wand"
        hit_id: Optional[int] = None

        if persons:
            for p in persons:
                p_id = p.get("id")
                if p_id == exclude_id: continue

                p_pos = p.get("pos")
                if p_pos is None: continue

                p_min = p_pos - np.array([25, 0, 25])
                p_max = p_pos + np.array([25, 180, 25])
                
                t = self._intersect_aabb(origin, direction, p_min, p_max)
                if t is not None and t < t_min:
                    t_min = t
                    hit_pos = origin + t * direction
                    hit_type = "person"
                    hit_id = p_id
                    hit_label = f"Person {p_id}"

        if room_objects:
            for obj in room_objects:
                if not obj.get("is_visible", True): continue
                
                pos: np.ndarray = np.array([float(obj.get("pos_x", 0)), float(obj.get("pos_y", 0)), float(obj.get("pos_z", 0))])
                size: np.ndarray = np.array([float(obj.get("size_w", 50)), float(obj.get("size_h", 50)), float(obj.get("size_d", 50))])
                yaw: float = float(np.deg2rad(obj.get("rotation_yaw", 0)))
                
                local_origin: np.ndarray = origin - (pos + np.array([0, size[1]*0.5, 0]))
                c, s = np.cos(-yaw), np.sin(-yaw)
                rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                
                local_origin = rot_mat @ local_origin
                local_direction = rot_mat @ direction
                
                half_size = size * 0.5
                t = self._intersect_aabb(local_origin, local_direction, -half_size, half_size)
                if t is not None and t < t_min:
                    t_min = t
                    hit_pos = origin + t * direction
                    hit_type = "object"
                    hit_label = obj.get("label", "Objekt")
                    hit_id = obj.get("id")

        if room_dims:
            w: float = float(room_dims.get("width",  600.0))
            h: float = float(room_dims.get("height", 250.0))
            d: float = float(room_dims.get("depth",  800.0))
            
            # Wir nutzen return_exit=True, da wir uns im Raum befinden und die Wände von INNEN treffen wollen
            t_wall = self._intersect_aabb(origin, direction, np.array([0, 0, 0]), np.array([w, h, d]), return_exit=True)
            if t_wall is not None and t_wall < t_min:
                t_min = t_wall
                hit_pos = origin + t_wall * direction
                hit_type = "wall"
                hit_label = "Wand/Boden"

        if hit_pos is None:
            hit_pos = origin + direction * 1000.0

        return {
            "pos": hit_pos,
            "type": hit_type,
            "label": hit_label,
            "id": hit_id,
            "dist": t_min if t_min != float('inf') else 1000.0
        }

    @staticmethod
    def _intersect_aabb(origin: np.ndarray, direction: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, return_exit: bool = False) -> Optional[float]:
        """Slab-Methode für Ray-AABB Schnitt."""
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / direction
            t1 = (aabb_min - origin) * inv_dir
            t2 = (aabb_max - origin) * inv_dir
            
        t_min = np.max(np.minimum(t1, t2))
        t_max = np.min(np.maximum(t1, t2))
        
        if t_min <= t_max and t_max >= 0:
            if return_exit:
                return t_max
            return t_min if t_min >= 0 else None
        return None


    @staticmethod
    def _parse_dict_or_bool(value: Any, default: dict[str, Any]) -> dict[str, Any]:
        """Normalisiert bool/dict Einstellungen auf ein einheitliches Dict-Format."""
        if isinstance(value, bool):
            return {**default, "master": value}
        return value if isinstance(value, dict) else default