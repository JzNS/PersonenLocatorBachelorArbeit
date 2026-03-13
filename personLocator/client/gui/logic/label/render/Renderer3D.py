import cv2
import numpy as np

from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.gui.logic.label.render.CalibrationRenderer import CalibrationRenderer, RenderColors


class Renderer3D:
    """Kümmert sich exklusiv um das Zeichnen der virtuellen 3D-Welt."""

    @staticmethod
    def render_3d_scene(world_points_3d, rotation_yaw, rotation_pitch, mesh_connections, angles_to_measure,
                        point_labels, person_results=None, room_dims=None, view_options=None,
                        zoom_level=1.0, pixel_points=None, camera_pos_label=None, custom_rectangles=None) -> np.ndarray:
        """Orchestriert das gesamte Rendering der virtuellen
        3D-Welt, einschließlich Raumgrenzen, Kameravirtualisierung,
        Kalibrierungsnetzen und dynamischen Personen-Skeletten."""

        view_opts = view_options or {}
        canvas, view_ctx = Renderer3D._setup_3d_canvas(world_points_3d, room_dims, rotation_yaw, rotation_pitch)
        if canvas is None: return np.zeros((1080, 1920, 3), dtype=np.uint8)

        # --- 1. Cam-Position ---
        cam_pt_2d = None
        if pixel_points and world_points_3d and len(pixel_points) >= 4:
            pose = GeometryMath.get_camera_pose(pixel_points, world_points_3d, custom_rectangles)
            if pose:
                rvec, tvec, _, _ = pose
                R, _ = cv2.Rodrigues(rvec)
                cam_pos_world = -np.dot(R.T, tvec).flatten()
                cam_pt_2d = GeometryMath.project_3d_to_2d(cam_pos_world, view_ctx['center'], view_ctx['rot_params'],
                                                          view_ctx['offset'], zoom_level)

                if view_opts.get("show_camera_world", True) and cam_pt_2d is not None:
                    Renderer3D._draw_3d_camera_icon(canvas, cam_pt_2d)

            Renderer3D._draw_calculated_camera_pose(canvas, pixel_points, world_points_3d, custom_rectangles,
                                                    view_ctx, zoom_level)

        # --- 2. RAUM-SETUP ---
        if room_dims:
            if view_opts.get("show_real_world", True):
                Renderer3D._draw_3d_room_frame(canvas, room_dims, view_ctx, zoom_level)

            if view_opts.get("show_floor_grid", True):
                Renderer3D._draw_3d_floor_grid(canvas, room_dims, view_ctx, zoom_level)

        if world_points_3d and view_opts.get("show_camera_world", True):
            Renderer3D._draw_3d_calibration_mesh(canvas, world_points_3d, mesh_connections, point_labels,
                                                 view_ctx, zoom_level)

        if custom_rectangles and view_opts.get("show_real_world", True):
            Renderer3D._draw_3d_custom_rectangles(canvas, custom_rectangles, view_ctx, zoom_level)

        # --- 3. Rays ---
        if pixel_points and world_points_3d:
            show_rays = view_opts.get("show_rays", True)
            Renderer3D._draw_3d_ransac_analysis(canvas, pixel_points, world_points_3d, custom_rectangles,
                                                view_ctx, zoom_level, show_rays=show_rays)

        # --- 4. PERSONEN ---
        if person_results:
            active_cam_pos = cam_pt_2d if view_opts.get("show_sightlines", True) else None
            Renderer3D._draw_3d_persons(canvas, person_results, view_ctx, zoom_level, pixel_points,
                                        world_points_3d, view_opts, active_cam_pos, custom_rectangles)

        return canvas

    @staticmethod
    def _setup_3d_canvas(world_points_3d, room_dims, yaw, pitch):
        """Initialisiert das leere 3D-Canvas und berechnet
        die notwendigen Projektionskontexte (Zentrum, Rotation, Offset)
        für die korrekte 3D-zu-2D-Transformation."""
        W_IMG, H_IMG = 1920, 1080
        canvas = np.zeros((H_IMG, W_IMG, 3), dtype=np.uint8)
        nodes = np.array(world_points_3d) if world_points_3d else np.array([])
        if len(nodes) > 0:
            center = np.mean(nodes, axis=0)
        elif room_dims:
            center = np.array([room_dims["width"] / 2, room_dims["height"] / 2, room_dims["depth"] / 2])
        else:
            return None, None
        return canvas, {'center': center, 'rot_params': {'yaw': np.radians(yaw), 'pitch': np.radians(pitch)},
                        'offset': (W_IMG // 2, H_IMG // 2)}

    @staticmethod
    def _draw_3d_room_frame(canvas, dims, ctx, zoom):
        """Zeichnet die äußeren Begrenzungslinien (Bounding Box)
        des physikalischen Raums basierend auf den konfigurierten Raumdimensionen."""
        W, H, D = dims["width"], dims["height"], dims["depth"]
        corners = [np.array([0, 0, 0]), np.array([W, 0, 0]), np.array([0, H, 0]), np.array([W, H, 0]),
                   np.array([0, 0, D]), np.array([W, 0, D]), np.array([0, H, D]), np.array([W, H, D])]
        proj = [GeometryMath.project_3d_to_2d(p, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for p in
                corners]
        for s, e in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(canvas, proj[s], proj[e], RenderColors.GRAY_DARK, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Room: {int(W)}x{int(H)}x{int(D)}cm", (40, 1040), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    RenderColors.GRAY_MID, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_3d_floor_grid(canvas, dims, ctx, zoom):
        """Generiert ein dreidimensionales,
        perspektivisch korrektes Bodenraster (Matrix-Gitter)
        zur visuellen Orientierung in der Raumtiefe."""
        W, D = int(dims.get("width", 600)), int(dims.get("depth", 800))
        for x in range(0, W + 50, 50):
            p1 = GeometryMath.project_3d_to_2d(np.array([x, 0.0, 0.0]), ctx['center'], ctx['rot_params'], ctx['offset'],
                                               zoom)
            p2 = GeometryMath.project_3d_to_2d(np.array([x, 0.0, float(D)]), ctx['center'], ctx['rot_params'],
                                               ctx['offset'], zoom)
            cv2.line(canvas, p1, p2, RenderColors.MATRIX_GREEN, 1, cv2.LINE_AA)
        for z in range(0, D + 50, 50):
            p1 = GeometryMath.project_3d_to_2d(np.array([0.0, 0.0, z]), ctx['center'], ctx['rot_params'], ctx['offset'],
                                               zoom)
            p2 = GeometryMath.project_3d_to_2d(np.array([float(W), 0.0, z]), ctx['center'], ctx['rot_params'],
                                               ctx['offset'], zoom)
            cv2.line(canvas, p1, p2, RenderColors.MATRIX_GREEN, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_3d_calibration_mesh(canvas, nodes, connections, point_labels, ctx, zoom):
        """Rendert die Haupt-Kalibrierungspunkte im 3D-Raum,
        verbindet diese zu einem Netz und blendet die echten
        Kantenlängen in Zentimetern ein."""
        projected = [GeometryMath.project_3d_to_2d(p, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for p in
                     nodes]
        for s, e in connections:
            if s < len(projected) and e < len(projected):
                cv2.line(canvas, projected[s], projected[e], RenderColors.GRAY_MID, 1, cv2.LINE_AA)
                dist_cm = np.linalg.norm(nodes[s] - nodes[e])
                if dist_cm > 0:
                    mid_x, mid_y = (projected[s][0] + projected[e][0]) // 2, (projected[s][1] + projected[e][1]) // 2
                    CalibrationRenderer.draw_hud_text(canvas, f"{dist_cm:.0f}cm", (mid_x - 20, mid_y),
                                                      RenderColors.GRAY_LIGHT, 0.7, 1)
        for i, pt in enumerate(projected):
            cv2.circle(canvas, pt, 8, RenderColors.RED, -1, cv2.LINE_AA)
            CalibrationRenderer.draw_hud_text(canvas, point_labels[i], (pt[0] - 25, pt[1] - 25), RenderColors.CYAN, 0.8,
                                              2)

    @staticmethod
    def _draw_3d_custom_rectangles(canvas, rectangles, ctx, zoom):
        """Projiziert benutzerdefinierte 3D-Rechtecke auf das
        Canvas und beschriftet deren Eckpunkte farblich passend
        zur jeweiligen Rechteck-Kategorie."""
        for rect in rectangles:
            if not rect.get("is_active", True): continue
            color = CalibrationRenderer._get_rectangle_color(rect.get("type"))
            valid_3d = []

            for c in rect.get("corners", []):
                try:
                    x, y, z = float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 0))
                    p3d = np.array([x, y, z])
                    p2d = GeometryMath.project_3d_to_2d(p3d, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
                    valid_3d.append((p2d, c.get("label", "")))
                except (ValueError, TypeError):
                    pass

            pts_2d = [p[0] for p in valid_3d]
            if len(pts_2d) > 1:
                pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                is_closed = (len(pts_2d) == 4)
                cv2.polylines(canvas, [pts_arr], isClosed=is_closed, color=color, thickness=2, lineType=cv2.LINE_AA)

            for i, (p, label) in enumerate(valid_3d):
                cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
                text = f"{rect.get('display_id', '?')}.{i + 1} {label}"
                CalibrationRenderer.draw_hud_text(canvas, text, (p[0] + 8, p[1] - 8), color, 0.45, 1)

    @staticmethod
    def _draw_3d_ransac_analysis(canvas, pixel_points, world_points, custom_rectangles, ctx, zoom, show_rays=True):
        """Visualisiert die Raycast-Präzision durch das Zeichnen
        gestrichelter Fehlerlinien zwischen idealen Weltkoordinaten
        und den tatsächlichen Echt-3D-Punkten."""
        if not show_rays: return
        if custom_rectangles:
            for rect in custom_rectangles:
                if not rect.get("is_active", True): continue
                color = CalibrationRenderer._get_rectangle_color(rect.get("type"))
                for corner in rect.get("corners", []):
                    px, py = corner.get("px"), corner.get("py")
                    if px is None or py is None: continue

                    ideal_p3d = np.array([corner["x"], corner["y"], corner["z"]])
                    real_p3d = GeometryMath.project_2d_to_3d(px, py, pixel_points, world_points, custom_rectangles,
                                                             target_y=ideal_p3d[1])

                    if real_p3d is not None:
                        p2d_real = GeometryMath.project_3d_to_2d(real_p3d, ctx['center'], ctx['rot_params'],
                                                                 ctx['offset'], zoom)
                        p2d_ideal = GeometryMath.project_3d_to_2d(ideal_p3d, ctx['center'], ctx['rot_params'],
                                                                  ctx['offset'], zoom)

                        err = np.linalg.norm(real_p3d - ideal_p3d)
                        CalibrationRenderer.draw_dashed_line(canvas, p2d_real, p2d_ideal, color, 1, dash_length=6)
                        cv2.circle(canvas, p2d_real, 3, color, -1, cv2.LINE_AA)
                        CalibrationRenderer.draw_hud_text(canvas, f"{err:.1f}cm", (p2d_real[0] + 5, p2d_real[1]), color,
                                                          0.4, 1)

    @staticmethod
    def _draw_3d_camera_icon(canvas, cam_pt_2d):
        """Zeichnet ein visuelles Kamerasymbol (Icon und Label)
         an der exakten berechneten Position der physischen
         Kamera innerhalb der virtuellen 3D-Welt."""
        cv2.circle(canvas, cam_pt_2d, 25, RenderColors.CYAN, -1, cv2.LINE_AA)
        cv2.circle(canvas, cam_pt_2d, 25, RenderColors.BLACK, 2, cv2.LINE_AA)
        CalibrationRenderer.draw_hud_text(canvas, "CAM", (cam_pt_2d[0] - 30, cam_pt_2d[1] - 35), RenderColors.CYAN, 1.0,
                                          2)

    @staticmethod
    def _draw_3d_persons(canvas, persons, ctx, zoom, pixel_points, world_points, view_options, cam_pos_2d,
                         custom_rectangles=None):
        """Koordiniert das Rendering erkannter Personen
        im 3D-Raum und wählt dynamisch zwischen detaillierter
        Skelett-Darstellung und einer einfachen Säulen-Visualisierung."""
        show_skel = view_options.get("show_skeleton_3d", True)

        for p in persons:
            base_3d = p['pos']
            color = RenderColors.PERSON_STAND
            skel_3d = {}

            if show_skel and len(pixel_points) >= 4 and len(world_points) >= 4:
                skel_3d = GeometryMath.lift_skeleton_to_3d(p.get('keypoints', []), base_3d, pixel_points, world_points,
                                                           custom_rectangles)

            if skel_3d:
                label_pos = Renderer3D._draw_person_skeleton(canvas, skel_3d, color, p.get('stored_leg_len', 0.0), ctx,
                                                             zoom, cam_pos_2d)
            else:
                label_pos = Renderer3D._draw_person_pillar(canvas, base_3d, p.get('height', 180), color, ctx, zoom,
                                                           cam_pos_2d)

            name_label = p.get('status', f"ID:{p['id']}")
            if label_pos is not None:
                CalibrationRenderer.draw_hud_text(canvas, f"{name_label} | {p.get('height', 0):.0f}cm",
                                                  (label_pos[0] + 15, label_pos[1] - 15), color, 0.8, 1)

    @staticmethod
    def _draw_person_skeleton(canvas, skel_3d, color, stored_leg_len, ctx, zoom, cam_pos_2d):
        """Projiziert die 3D-Gelenkkoordinaten,
        verbindet diese zu einem anatomischen Skelett
        und extrapoliert bei Verdeckungen fehlende Beinstrukturen zum Boden."""
        projected_joints = {
            j_id: GeometryMath.project_3d_to_2d(pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for
            j_id, pos in skel_3d.items()}

        Renderer3D._draw_sightlines(canvas, projected_joints.values(), cam_pos_2d)

        for pt_2d in projected_joints.values():
            cv2.circle(canvas, pt_2d, 4, color, -1, cv2.LINE_AA)

        for s, e in CalibrationRenderer.SKELETON_LINKS:
            if s in projected_joints and e in projected_joints:
                cv2.line(canvas, projected_joints[s], projected_joints[e], RenderColors.GRAY_LIGHT, 2, cv2.LINE_AA)

        if not (15 in skel_3d or 16 in skel_3d) and (11 in skel_3d and 12 in skel_3d) and stored_leg_len > 0:
            hip_center = (skel_3d[11] + skel_3d[12]) / 2
            p_hip = GeometryMath.project_3d_to_2d(hip_center, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
            p_foot = GeometryMath.project_3d_to_2d(np.array([hip_center[0], 0, hip_center[2]]), ctx['center'],
                                                   ctx['rot_params'], ctx['offset'], zoom)
            cv2.line(canvas, p_hip, p_foot, RenderColors.GRAY_DARK, 2, cv2.LINE_AA)
            cv2.circle(canvas, p_foot, 6, color, -1, cv2.LINE_AA)

        head_id = 0 if 0 in skel_3d else (5 if 5 in skel_3d else -1)
        return projected_joints[head_id] if head_id != -1 else (
            list(projected_joints.values())[0] if projected_joints else None)

    @staticmethod
    def _draw_person_pillar(canvas, base_3d, height, color, ctx, zoom, cam_pos_2d):
        """Dient als Fallback-Visualisierung und zeichnet eine Person als simplen,
        vertikalen Vektor (Säule) von der Bodenposition bis zur geschätzten Kopfhöhe."""
        if base_3d is None: return None
        b2d = GeometryMath.project_3d_to_2d(base_3d, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)
        t2d = GeometryMath.project_3d_to_2d(base_3d + np.array([0, height, 0]), ctx['center'], ctx['rot_params'],
                                            ctx['offset'], zoom)
        cv2.line(canvas, b2d, t2d, color, 3, cv2.LINE_AA)
        Renderer3D._draw_sightlines(canvas, [t2d], cam_pos_2d)
        return t2d

    @staticmethod
    def _draw_sightlines(canvas, target_points_2d, cam_pos_2d):
        """Generiert semitransparente Sichtlinien (Rays)
        von der virtuellen Kameraposition zu spezifischen
        Zielpunkten wie den Köpfen der erfassten Personen."""
        if cam_pos_2d is None or not target_points_2d: return
        overlay = canvas.copy()
        for pt in target_points_2d:
            cv2.line(overlay, cam_pos_2d, pt, RenderColors.SIGHTLINE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    @staticmethod
    def _draw_calculated_camera_pose(canvas: np.ndarray, pixel_points: list, world_points: list,
                                     custom_rectangles: list, ctx: dict, zoom: float) -> None:
        """Extrahiert die exakten physikalischen X/Y/Z-Koordinaten
        sowie die Pitch/Yaw/Roll-Winkel aus der PnP-Pose und rendert diese
        als übersichtliches HUD-Element."""
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles, img_size=(1920, 1080))
        if not pose: return

        rvec, tvec, _, _ = pose
        R, _ = cv2.Rodrigues(rvec)
        cam_pos_world = -np.dot(R.T, tvec).flatten()

        sy = np.sqrt(R.T[0, 0] ** 2 + R.T[1, 0] ** 2)
        pitch = np.arctan2(R.T[2, 1], R.T[2, 2]) if sy > 1e-6 else np.arctan2(-R.T[1, 2], R.T[1, 1])
        yaw = np.arctan2(-R.T[2, 0], sy)
        roll = np.arctan2(R.T[1, 0], R.T[0, 0]) if sy > 1e-6 else 0

        deg = np.degrees([pitch, yaw, roll])
        x, y, z = cam_pos_world

        hud_x, hud_y = canvas.shape[1] - 460, 70
        lines = [
            "--- BERECHNETE KAMERA ---",
            f"X: {x:8.1f} cm", f"Y: {y:8.1f} cm", f"Z: {z:8.1f} cm",
            f"Pitch: {deg[0]:5.1f} deg", f"Yaw:   {deg[1]:5.1f} deg", f"Roll:  {deg[2]:5.1f} deg"
        ]

        for i, line in enumerate(lines):
            color = RenderColors.MAGENTA if i == 0 else RenderColors.WHITE
            CalibrationRenderer.draw_hud_text(canvas, line, (hud_x, hud_y + i * 45), color, 0.95, 2)