import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from server.core.math.GeometryMath import GeometryMath


class DrawingUtils:
    """
    Rein statische Zeichen-Helfer für OpenCV-Canvas.
    Keine Zustandsvariablen – alle Methoden sind zustandslos und thread-sicher.
    """

    @staticmethod
    def draw_hud_text(
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.8,
        thickness: int = 2,
    ) -> None:
        """Zeichnet Text mit schwarzem Hintergrund-Rechteck auf den Canvas."""
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        rect_start: tuple[int, int] = (int(pos[0] - 8), int(pos[1] - h - 12))
        rect_end: tuple[int, int] = (int(pos[0] + w + 8), int(pos[1] + 8))
        cv2.rectangle(img, rect_start, rect_end, (0, 0, 0), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_room_frame(
        canvas: np.ndarray,
        dims: dict[str, float],
        ctx: dict[str, Any],
        zoom: float,
    ) -> None:
        """Zeichnet das 3D-Raumrahmen-Drahtgitter auf den Canvas."""
        width: float  = float(dims.get("width",  600.0))
        height: float = float(dims.get("height", 250.0))
        depth: float  = float(dims.get("depth",  800.0))

        corners: list[np.ndarray] = [
            np.array([0,     0,      0], dtype=np.float32),      np.array([width, 0,      0], dtype=np.float32),
            np.array([0,     height, 0], dtype=np.float32),      np.array([width, height, 0], dtype=np.float32),
            np.array([0,     0,      depth], dtype=np.float32),  np.array([width, 0,      depth], dtype=np.float32),
            np.array([0,     height, depth], dtype=np.float32),  np.array([width, height, depth], dtype=np.float32),
        ]
        proj: list[tuple[int, int]] = [
            GeometryMath.project_3d_to_2d(p, ctx["center"], ctx["rot_params"], ctx["offset"], zoom)
            for p in corners
        ]
        edges: list[tuple[int, int]] = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
        for s, e in edges:
            cv2.line(canvas, proj[s], proj[e], (80, 80, 80), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"Raum: {int(width)}x{int(height)}x{int(depth)}cm",
            (40, 1040),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA,
        )

    @staticmethod
    def draw_custom_rectangles(
        canvas: np.ndarray,
        rectangles: list[dict[str, Any]],
        ctx: dict[str, Any],
        zoom: float,
    ) -> None:
        """Zeichnet alle aktiven benutzerdefinierten Rechtecke in der 3D-Ansicht."""
        for rect in rectangles:
            if not rect.get("is_active", True):
                continue
            rect_pts: list[tuple[int, int]] = []
            for c in rect.get("corners", []):
                corner_3d: np.ndarray = np.array([float(c["x"]), float(c["y"]), float(c["z"])], dtype=np.float32)
                pt_2d: tuple[int, int] = GeometryMath.project_3d_to_2d(
                    corner_3d, ctx["center"], ctx["rot_params"], ctx["offset"], zoom
                )
                if pt_2d:
                    rect_pts.append(pt_2d)
            if len(rect_pts) == 4:
                cv2.polylines(canvas, [np.array(rect_pts, dtype=np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
                label_pos: tuple[int, int] = (rect_pts[0][0], rect_pts[0][1] - 10)
                DrawingUtils.draw_hud_text(
                    canvas, str(rect.get("display_id", "Viereck")), label_pos, (0, 255, 255), 0.5, 1
                )

    @staticmethod
    def draw_room_objects(
        canvas: np.ndarray,
        room_objects: list[dict[str, Any]],
        ctx: dict[str, Any],
        zoom: float,
    ) -> None:
        """
        Zeichnet platzierte Raum-Objekte als Drahtgitter-Quader (mit Yaw-Rotation und farbiger Boden-Markierung).

        Konvention: (pos_x, pos_y, pos_z) ist der Mittelpunkt der Bodenfläche.
        Das Objekt erstreckt sich um size_w/2 in X und size_d/2 in Z um diesen Mittelpunkt
        und um size_h vertikal in Y (vom Boden nach oben).
        """
        if not room_objects:
            return

        edges: list[tuple[int, int]] = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for obj in room_objects:
            if not obj.get("is_visible", True):
                continue

            px: float = float(obj.get("pos_x", 0.0))
            py: float = float(obj.get("pos_y", 0.0))
            pz: float = float(obj.get("pos_z", 0.0))
            sw: float = max(1.0, float(obj.get("size_w", 50.0)))
            sh: float = max(1.0, float(obj.get("size_h", 50.0)))
            sd: float = max(1.0, float(obj.get("size_d", 50.0)))
            yaw_deg: float = float(obj.get("rotation_yaw", 0.0))

            color_hex: str = str(obj.get("color_hex", "#FFAA00")).lstrip("#")
            try:
                r: int = int(color_hex[0:2], 16)
                g: int = int(color_hex[2:4], 16)
                b: int = int(color_hex[4:6], 16)
                color_bgr: tuple[int, int, int] = (b, g, r)
            except Exception:
                color_bgr = (0, 170, 255)

            half_w: float = sw * 0.5
            half_d: float = sd * 0.5
            local_corners: list[tuple[float, float, float]] = [
                (-half_w, 0.0,  -half_d), ( half_w, 0.0,  -half_d),
                (-half_w, 0.0,   half_d), ( half_w, 0.0,   half_d),
                (-half_w, sh,   -half_d), ( half_w, sh,   -half_d),
                (-half_w, sh,    half_d), ( half_w, sh,    half_d),
            ]

            yaw_rad: float = float(np.deg2rad(yaw_deg))
            cos_y: float = float(np.cos(yaw_rad))
            sin_y: float = float(np.sin(yaw_rad))

            proj: list[tuple[int, int]] = []
            for lx, ly, lz in local_corners:
                wx: float = lx * cos_y - lz * sin_y + px
                wy: float = ly + py
                wz: float = lx * sin_y + lz * cos_y + pz
                pt_2d: tuple[int, int] = GeometryMath.project_3d_to_2d(
                    np.array([wx, wy, wz], dtype=np.float32),
                    ctx["center"], ctx["rot_params"], ctx["offset"], zoom,
                )
                if pt_2d:
                    proj.append(pt_2d)
                else:
                    proj.append((0, 0))

            if len(proj) != 8:
                continue

            try:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.array([proj[0], proj[1], proj[3], proj[2]], dtype=np.int32)], color_bgr)
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)
            except Exception:
                pass

            for s, e in edges:
                cv2.line(canvas, proj[s], proj[e], color_bgr, 2, cv2.LINE_AA)

            front_x: float = px + sin_y * (half_d + 15.0)
            front_z: float = pz + cos_y * (half_d + 15.0)
            base_2d: tuple[int, int] = GeometryMath.project_3d_to_2d(
                np.array([px, py, pz], dtype=np.float32),
                ctx["center"], ctx["rot_params"], ctx["offset"], zoom,
            )
            front_2d: tuple[int, int] = GeometryMath.project_3d_to_2d(
                np.array([front_x, py, front_z], dtype=np.float32),
                ctx["center"], ctx["rot_params"], ctx["offset"], zoom,
            )
            if base_2d and front_2d:
                cv2.arrowedLine(canvas, base_2d, front_2d, color_bgr, 2, cv2.LINE_AA, tipLength=0.4)

            label: str = str(obj.get("name", "Objekt"))
            label_anchor: tuple[int, int] = (proj[4][0], proj[4][1] - 10)
            DrawingUtils.draw_hud_text(canvas, label, label_anchor, color_bgr, 0.55, 1)

    @staticmethod
    def draw_cameras(
        canvas: np.ndarray,
        all_cams_2d: dict[str, tuple[int, int]],
    ) -> None:
        """Zeichnet alle Kamera-Symbole mit farbigen Kreisen und Labels."""
        cam_colors: dict[str, tuple[int, int, int]] = {
            "CAMERA_1": (80,  80,  255),
            "CAMERA_2": (255, 150, 80),
            "CAMERA_3": (80,  255, 80),
            "CAMERA_4": (80,  255, 255),
        }
        for cam_name, pt_2d in all_cams_2d.items():
            color: tuple[int, int, int] = cam_colors.get(cam_name, (255, 200, 0))
            cv2.circle(canvas, pt_2d, 35, color,  -1, cv2.LINE_AA)
            cv2.circle(canvas, pt_2d, 35, (0,0,0), 4, cv2.LINE_AA)
            display_name: str = cam_name.replace("CAMERA_", "CAM ") if "CAMERA_" in cam_name else "KAMERA"
            DrawingUtils.draw_hud_text(canvas, display_name, (pt_2d[0] - 45, pt_2d[1] - 50), color, 1.0, 2)

    @staticmethod
    def calculate_epipolar_distance_batch(
        orig1: np.ndarray,
        dir1:  np.ndarray,
        orig2: np.ndarray,
        dir2:  np.ndarray,
    ) -> np.ndarray:
        """
        Vektorisierte Epipolar-Distanz für N Gelenke GLEICHZEITIG.
        Input Shape: (N, 3) — gibt (N,)-Array mit Distanzen zurück.
        """
        cross_vec: np.ndarray   = np.cross(dir1, dir2)
        denominator: np.ndarray = np.sum(cross_vec ** 2, axis=1)
        parallel_mask: np.ndarray = denominator < 1e-6

        diff: np.ndarray = orig2 - orig1
        dot_diff_cross: np.ndarray = np.einsum("ij,ij->i", diff, cross_vec)
        safe_norm: np.ndarray  = np.where(parallel_mask, 1.0, np.sqrt(denominator))
        distances: np.ndarray  = np.abs(dot_diff_cross) / safe_norm

        cross_diff_dir2: np.ndarray = np.cross(diff, dir2)
        t1: np.ndarray = np.einsum("ij,ij->i", cross_diff_dir2, cross_vec) / np.where(parallel_mask, 1.0, denominator)
        cross_diff_dir1: np.ndarray = np.cross(diff, dir1)
        t2: np.ndarray = np.einsum("ij,ij->i", cross_diff_dir1, cross_vec) / np.where(parallel_mask, 1.0, denominator)

        behind_mask: np.ndarray = (t1 < 0) | (t2 < 0)
        distances[behind_mask & ~parallel_mask] = 9999.0

        if np.any(parallel_mask):
            diff_p: np.ndarray = diff[parallel_mask]
            dir1_p: np.ndarray = dir1[parallel_mask]
            dot_diff_dir1: np.ndarray = np.einsum("ij,ij->i", diff_p, dir1_p)
            proj: np.ndarray = diff_p - dot_diff_dir1[:, np.newaxis] * dir1_p
            distances[parallel_mask] = np.linalg.norm(proj, axis=1)

        return distances

    @staticmethod
    def calculate_color_distance(
        colors_a: dict[Union[int, str], str],
        colors_b: dict[Union[int, str], str],
    ) -> float:
        """Normierter RGB-Abstand zweier Gelenk-Farbmengen (0 = identisch, 1 = maximal verschieden)."""
        if not colors_a or not colors_b:
            return 1.0
        common: Set[Union[int, str]] = set(colors_a).intersection(colors_b)
        if not common:
            return 1.0

        diffs: list[float] = []
        for j_id in common:
            try:
                c1: str = colors_a[j_id]
                c2: str = colors_b[j_id]
                rgb1: np.ndarray = np.array([int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)])
                rgb2: np.ndarray = np.array([int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)])
                diffs.append(float(np.linalg.norm(rgb1 - rgb2) / 441.67))
            except (ValueError, IndexError):
                continue

        return float(np.mean(diffs)) if diffs else 1.0