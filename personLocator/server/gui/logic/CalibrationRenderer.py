import cv2
import numpy as np
from server.gui.logic.GeometryMath import GeometryMath

class CalibrationRenderer:
    """Übernimmt die Visualisierung (2D Overlay und 3D Vorschau) auf dem Server."""

    SKELETON_LINKS = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    @staticmethod
    def draw_hud_text(img, text, pos, color, scale=0.8, thickness=2):
        """Zeichnet lesbaren Text mit schwarzem Hintergrund für bessere Sichtbarkeit."""
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (int(pos[0] - 8), int(pos[1] - h - 12)), (int(pos[0] + w + 8), int(pos[1] + 8)), (0, 0, 0), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8):
        """Zeichnet eine gestrichelte Linie zwischen zwei Punkten."""
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0: return
        dashes = max(1, int(dist / dash_length))
        for i in range(dashes):
            start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes))
            cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

    @staticmethod
    def render_3d_scene(world_points_3d, rotation_yaw, rotation_pitch, mesh_connections, angles_to_measure,
                        point_labels, person_results=None, room_dims=None, view_options=None,
                        zoom_level=1.0, pixel_points=None, camera_pos_label=None,
                        custom_rectangles=None, img_size=(1920, 1080), camera_matrix=None, dist_coeffs=None):
        """Hauptmethode zum Rendern der 3D-Szene mit allen Elementen (Raum, Kamera, Personen, Vierecke)."""

        W_IMG, H_IMG = 1920, 1080
        canvas = np.zeros((H_IMG, W_IMG, 3), dtype=np.uint8)
        if view_options is None: view_options = {"show_real_world": True, "show_camera_world": True}

        # 1. Kamera Position (PnP) absolut sauber berechnen
        cam_pos_world = None
        pose = GeometryMath.get_camera_pose(pixel_points, world_points_3d, custom_rectangles, img_size, camera_matrix, dist_coeffs)
        if pose:
            rvec, tvec, K, dist = pose
            R, _ = cv2.Rodrigues(rvec)
            cam_pos_world = -np.dot(R.T, tvec).flatten()

        # 2. Anti-Clipping Bounding Box spannen
        nodes = []
        if room_dims:
            W, H, D = room_dims.get("width", 600), room_dims.get("height", 250), room_dims.get("depth", 800)
            nodes.extend([[0, 0, 0], [W, 0, 0], [0, 0, D], [W, 0, D], [0, H, 0], [W, H, D]])
        if cam_pos_world is not None:
            nodes.append(cam_pos_world)

        if len(nodes) > 0:
            center = np.mean(nodes, axis=0)
        else:
            return canvas

        view_ctx = {
            'center': center,
            'rot_params': {'yaw': np.radians(rotation_yaw), 'pitch': np.radians(rotation_pitch)},
            'offset': (W_IMG // 2, H_IMG // 2)
        }

        # 3. Kamera 2D Punkt berechnen
        cam_pt_2d = None
        if cam_pos_world is not None:
            cam_pt_2d = GeometryMath.project_3d_to_2d(cam_pos_world, view_ctx['center'], view_ctx['rot_params'], view_ctx['offset'], zoom_level)

        # 4. Den physischen Raum sauber zeichnen
        if room_dims and view_options.get("show_real_world", True):
            CalibrationRenderer.draw_room_frame(canvas, room_dims, view_ctx, zoom_level)

        # 5. Die PnP Vierecke auf dem virtuellen Boden visualisieren
        if custom_rectangles and view_options.get("show_real_world", True):
            for rect in custom_rectangles:
                if not rect.get("is_active", True): continue
                rect_pts = []
                for c in rect.get("corners", []):
                    p3d = np.array([float(c["x"]), float(c["y"]), float(c["z"])])
                    p2d = GeometryMath.project_3d_to_2d(p3d, view_ctx['center'], view_ctx['rot_params'], view_ctx['offset'], zoom_level)
                    rect_pts.append(p2d)

                if len(rect_pts) == 4:
                    cv2.polylines(canvas, [np.array(rect_pts, dtype=np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
                    CalibrationRenderer.draw_hud_text(canvas, rect.get("display_id", "Viereck"), (rect_pts[0][0], rect_pts[0][1] - 10), (0, 255, 255), 0.5, 1)

        # 6. Personen & Skelette zeichnen
        if person_results:
            CalibrationRenderer.draw_persons_3d(
                canvas, person_results, view_ctx, zoom_level,
                pixel_points, world_points_3d, view_options, cam_pt_2d,
                custom_rectangles=custom_rectangles,
                img_size=img_size, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs # <--- FIX: Weitergabe
            )

        # 7. Kamera Icon zeichnen
        if cam_pt_2d is not None and view_options.get("show_camera_world", True):
            cv2.circle(canvas, cam_pt_2d, 35, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, cam_pt_2d, 35, (0, 0, 0), 4, cv2.LINE_AA)
            CalibrationRenderer.draw_hud_text(canvas, "KAMERA", (cam_pt_2d[0] - 45, cam_pt_2d[1] - 50), (255, 255, 0), 1.0, 2)

        return canvas

    @staticmethod
    def draw_persons_3d(canvas, persons, ctx, zoom, pixel_points, world_points, view_options=None, cam_pos_2d=None,
                        custom_rectangles=None, img_size=(1920, 1080), camera_matrix=None, dist_coeffs=None):
        """Zeichnet die Personen mit optionalen 3D-Skeletten, Höhenangaben und Status-Labels auf dem Canvas."""
        show_skel = view_options.get("show_skeleton_3d", True) if view_options else True

        for p in persons:
            base_3d = p['pos']
            keypoints = p.get('keypoints', [])
            color = (0, 255, 127) if "Stehend" in p.get('status', '') else (0, 165, 255)

            skel_3d = {}
            if show_skel and hasattr(GeometryMath, 'lift_skeleton_to_3d'):
                try:
                    skel_3d = GeometryMath.lift_skeleton_to_3d(keypoints, base_3d, pixel_points, world_points,
                                                               custom_rectangles, img_size, camera_matrix, dist_coeffs)
                except Exception:
                    pass

            projected_joints = {}
            if skel_3d:
                for j_id, pos in skel_3d.items():
                    projected_joints[j_id] = GeometryMath.project_3d_to_2d(pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            if cam_pos_2d is not None and projected_joints:
                overlay = canvas.copy()
                for j_id, pt_2d in projected_joints.items():
                    cv2.line(overlay, cam_pos_2d, pt_2d, (200, 255, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

            if skel_3d:
                for pt_2d in projected_joints.values():
                    cv2.circle(canvas, pt_2d, 6, color, -1, cv2.LINE_AA)

                for s, e in CalibrationRenderer.SKELETON_LINKS:
                    if s in projected_joints and e in projected_joints:
                        cv2.line(canvas, projected_joints[s], projected_joints[e], (200, 200, 200), 3, cv2.LINE_AA)

                if 11 in skel_3d and 12 in skel_3d:
                    hip_center = (skel_3d[11] + skel_3d[12]) / 2
                    p_hip = GeometryMath.project_3d_to_2d(hip_center, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
                    ground_pos = np.array([hip_center[0], 0.0, hip_center[2]])
                    p_foot = GeometryMath.project_3d_to_2d(ground_pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

                    CalibrationRenderer.draw_dashed_line(canvas, p_hip, p_foot, (100, 100, 100), 2, 6)
                    cv2.ellipse(canvas, p_foot, (20, 8), 0, 0, 360, color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, p_foot, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                h = p.get('height', 180)
                b2d = GeometryMath.project_3d_to_2d(base_3d, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
                t2d = GeometryMath.project_3d_to_2d(base_3d + np.array([0, h, 0]), ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
                cv2.line(canvas, b2d, t2d, color, 5, cv2.LINE_AA)
                cv2.ellipse(canvas, b2d, (20, 8), 0, 0, 360, color, 2, cv2.LINE_AA)

                if cam_pos_2d is not None:
                    overlay = canvas.copy()
                    cv2.line(overlay, cam_pos_2d, t2d, (200, 255, 255), 2, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

            label_pos = None
            if skel_3d:
                head_id = 0 if 0 in skel_3d else (5 if 5 in skel_3d else -1)
                if head_id != -1: label_pos = projected_joints[head_id]

            if label_pos is None:
                label_pos = GeometryMath.project_3d_to_2d(base_3d + np.array([0, p.get('height', 180), 0]),
                                                          ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            name_label = p.get('status', f"ID:{p['id']}")
            CalibrationRenderer.draw_hud_text(canvas, f"{name_label} | {p.get('height', 0):.0f}cm",
                                              (int(label_pos[0] + 15), int(label_pos[1] - 15)), color, 0.9, 2)

    @staticmethod
    def draw_room_frame(canvas, dims: dict, ctx, zoom):
        """Zeichnet den 3D-Rahmen des Raumes basierend auf den angegebenen Dimensionen."""
        W, H, D = dims.get("width", 600), dims.get("height", 250), dims.get("depth", 800)
        corners = [np.array([0, 0, 0]), np.array([W, 0, 0]), np.array([0, H, 0]), np.array([W, H, 0]),
                   np.array([0, 0, D]), np.array([W, 0, D]), np.array([0, H, D]), np.array([W, H, D])]

        proj = [GeometryMath.project_3d_to_2d(p, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for p in corners]
        lines = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for s, e in lines:
            cv2.line(canvas, proj[s], proj[e], (80, 80, 80), 2, cv2.LINE_AA)

        cv2.putText(canvas, f"Raum: {int(W)}x{int(H)}x{int(D)}cm", (40, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA)