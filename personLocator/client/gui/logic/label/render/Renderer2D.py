import cv2
import numpy as np

from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.gui.logic.label.render.CalibrationRenderer import CalibrationRenderer, RenderColors

class Renderer2D:

    @staticmethod
    def render_2d_toolbox(frame: np.ndarray, scale: float,
                          dead_zones: list, mirror_zones: list,
                          pixel_points: list, custom_rectangles: list,
                          mesh_connections: list, world_points: list,
                          room_dims: dict, view_options: dict) -> np.ndarray:
        """Skaliert und zeichnet das vollständige 2D-Overlay auf das Kamerabild,
        inklusive der visuellen Hervorhebung von Dead- und Mirror-Zonen sowie
        dynamisch angepasster Custom-Vierecke."""
        if frame is None: return frame

        def scale_points(points):
            if scale == 1.0: return np.array(points, np.int32).reshape((-1, 1, 2))
            return (np.array(points) * scale).astype(np.int32).reshape((-1, 1, 2))

        # 1. Zonen zeichnen
        if dead_zones:
            ov = frame.copy()
            for p in dead_zones: cv2.fillPoly(ov, [scale_points(p)], (0, 0, 255))
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)

        if mirror_zones:
            ov = frame.copy()
            for p in mirror_zones: cv2.fillPoly(ov, [scale_points(p)], (255, 0, 0))
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)

        scaled_pixels = [(int(x * scale), int(y * scale)) for x, y in pixel_points]

        scaled_custom_rects = []
        for rect in custom_rectangles:
            if not rect.get("is_active", True):
                continue
            if any(c.get("px") is None or c.get("py") is None for c in rect.get("corners", [])):
                continue

            scaled_rect = dict(rect)
            scaled_corners = []
            for c in rect.get("corners", []):
                sc = dict(c)
                sc["px"] = int(sc["px"] * scale)
                sc["py"] = int(sc["py"] * scale)
                scaled_corners.append(sc)

            scaled_rect["corners"] = scaled_corners
            scaled_custom_rects.append(scaled_rect)

        return Renderer2D.draw_2d_overlay(
            frame=frame,
            pixel_points=scaled_pixels,
            connections=mesh_connections,
            custom_rectangles=scaled_custom_rects,
            world_points=world_points,
            room_dims=room_dims,
            view_options=view_options
        )

    @staticmethod
    def draw_2d_overlay(frame: np.ndarray, pixel_points: list, connections: list,
                        custom_rectangles: list = None,
                        world_points: list = None,
                        room_dims: dict = None,
                        view_options: dict = None) -> np.ndarray:
        """Fungiert als zentrale Rendering-Pipeline für das 2D-Kamerabild,
        die basierend auf den übergebenen Ansichtsoptionen Kalibrierungspunkte,
        Gitterlinien und benutzerdefinierte Rechtecke zusammensetzt."""
        canvas = frame.copy()
        view_opts = view_options or {"show_real_world": True, "show_camera_world": True}

        if view_opts.get("show_camera_world", True):
            Renderer2D._draw_2d_calibration_points(canvas, pixel_points)
            Renderer2D._draw_2d_grid(canvas, pixel_points, connections)

        if custom_rectangles and view_opts.get("show_real_world", True):
            Renderer2D._draw_2d_custom_rectangles(canvas, custom_rectangles)

        return canvas

    @staticmethod
    def _draw_2d_calibration_points(canvas, pixel_points):
        """Zeichnet die primären Raum-Kalibrierungspunkte
        auf das Canvas und versieht diese zur besseren
        Lesbarkeit mit kontrastreichen, textbasierten Raum-Labels
        (z.B. HUL, HUR) ohne störende Hintergrundboxen."""
        labels = ["HUL", "HUR", "HOL", "HOR", "VUL", "VOL", "VUR", "VOR"]

        for i, pt in enumerate(pixel_points):
            if pt == (0, 0) and i > 0: continue

            color = RenderColors.YELLOW if i < 4 else RenderColors.RED
            cv2.circle(canvas, pt, 5, color, -1, cv2.LINE_AA)

            label_text = labels[i] if i < len(labels) else str(i)

            cv2.putText(canvas, label_text, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RenderColors.BLACK,
                        2, cv2.LINE_AA)
            cv2.putText(canvas, label_text, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

    @staticmethod
    def _draw_2d_grid(canvas, pixel_points, connections):
        """Verbindet die definierten Kalibrierungspunkte anhand
        der übergebenen Kanten-Matrix mit Linien,
        um das grundlegende Raumgitter im 2D-Bild visuell zu verdeutlichen."""
        if len(pixel_points) < 2: return
        for s, e in connections:
            if s < len(pixel_points) and e < len(pixel_points):
                p1, p2 = pixel_points[s], pixel_points[e]
                if p1 != (0, 0) and p2 != (0, 0):
                    cv2.line(canvas, p1, p2, RenderColors.GREEN, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_2d_custom_rectangles(canvas, custom_rectangles):
        """Iteriert über alle aktiven, benutzerdefinierten Rechtecke
        und rendert deren validierte Eckpunkte sowie die verbindenden
        Polygonlinien in der entsprechend zugewiesenen Kategorie-Farbe."""
        for rect in custom_rectangles:
            if not rect.get("is_active", True): continue
            color = CalibrationRenderer._get_rectangle_color(rect.get("type"))
            valid_pts = []

            for c in rect.get("corners", []):
                if c.get("px") is not None and c.get("py") is not None:
                    try:
                        px, py = int(c["px"]), int(c["py"])
                        valid_pts.append(((px, py), c.get("label", "")))
                    except ValueError:
                        pass

            pts_2d = [p[0] for p in valid_pts]
            if len(pts_2d) > 1:
                pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                is_closed = (len(pts_2d) == 4)
                cv2.polylines(canvas, [pts_arr], isClosed=is_closed, color=color, thickness=2, lineType=cv2.LINE_AA)

            for i, (p, label) in enumerate(valid_pts):
                cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_2d_matrix_grid(canvas, pixel_points, world_points, custom_rectangles, room_dims):
        """Projiziert ein dreidimensionales Bodenraster mittels der PnP-Kamerapose
        auf das zweidimensionale Live-Kamerabild,
        inklusive einer Filterung für Out-of-Bounds-Linien."""
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles,
                                            img_size=(canvas.shape[1], canvas.shape[0]))
        if pose is None: return

        rvec, tvec, K, dist_coeffs = pose
        W = int(room_dims.get("width", 600))
        D = int(room_dims.get("depth", 800))
        step = 50

        lines_3d = []
        for x in range(0, W + step, step):
            for z in range(0, D, step):
                lines_3d.append([[x, 0.0, z], [x, 0.0, z + step]])
        for z in range(0, D + step, step):
            for x in range(0, W, step):
                lines_3d.append([[x, 0.0, z], [x + step, 0.0, z]])

        if not lines_3d: return
        pts_3d_arr = np.array(lines_3d, dtype=np.float32).reshape(-1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
        pts_2d = pts_2d.reshape(-1, 2, 2)

        h_img, w_img = canvas.shape[:2]
        def is_valid(pt): return -w_img * 2 < pt[0] < w_img * 3 and -h_img * 2 < pt[1] < h_img * 3

        for p1, p2 in pts_2d:
            if is_valid(p1) and is_valid(p2):
                if np.linalg.norm(p1 - p2) < max(w_img, h_img):
                    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                             RenderColors.MATRIX_GREEN, 1, cv2.LINE_AA)