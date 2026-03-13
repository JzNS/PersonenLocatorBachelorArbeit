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

        view_opts = view_options or {}

        # --- 1. PnP Cam Position berechnen ---
        cam_pos_world = None
        if pixel_points and world_points_3d and len(pixel_points) >= 4:
            pose = GeometryMath.get_camera_pose(pixel_points, world_points_3d, custom_rectangles)
            if pose:
                rvec, tvec, _, _ = pose
                R, _ = cv2.Rodrigues(rvec)
                cam_pos_world = -np.dot(R.T, tvec).flatten()

        # --- 2. Canvas Setup ---
        canvas, view_ctx = Renderer3D._setup_3d_canvas(world_points_3d, room_dims, rotation_yaw, rotation_pitch,
                                                       cam_pos_world)
        if canvas is None: return np.zeros((1080, 1920, 3), dtype=np.uint8)

        # --- 3. Cam zeichnen ---
        cam_pt_2d = None
        if cam_pos_world is not None:
            cam_pt_2d = GeometryMath.project_3d_to_2d(cam_pos_world, view_ctx['center'], view_ctx['rot_params'],
                                                      view_ctx['offset'], zoom_level)
            if view_opts.get("show_camera_world", True) and cam_pt_2d is not None:
                Renderer3D._draw_3d_camera_icon(canvas, cam_pt_2d)
            Renderer3D._draw_calculated_camera_pose(canvas, pixel_points, world_points_3d, custom_rectangles, view_ctx,
                                                    zoom_level)

        # --- 4. Raum Setup ---
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
            show_rays = view_opts.get("show_rays", True)
            Renderer3D._draw_3d_ransac_analysis(canvas, pixel_points, world_points_3d, custom_rectangles, view_ctx,
                                                zoom_level, show_rays=show_rays)

        # --- 5. Person ---
        if person_results:
            active_cam_pos = cam_pt_2d if view_opts.get("show_sightlines", True) else None
            Renderer3D._draw_3d_persons(canvas, person_results, view_ctx, zoom_level, pixel_points, world_points_3d,
                                        view_opts, active_cam_pos, custom_rectangles)

        return canvas

    @staticmethod
    def _setup_3d_canvas(world_points_3d, room_dims, yaw, pitch, cam_pos_world=None):
        """Initialisiert das Canvas und berechnet den Mittelpunkt der Szene für die Projektion."""
        W_IMG, H_IMG = 1920, 1080
        canvas = np.zeros((H_IMG, W_IMG, 3), dtype=np.uint8)

        nodes = []
        if world_points_3d:
            nodes.extend(world_points_3d)

        if room_dims:
            W, H, D = room_dims.get("width", 600), room_dims.get("height", 250), room_dims.get("depth", 800)
            nodes.extend([[0, 0, 0], [W, 0, 0], [0, 0, D], [W, 0, D], [0, H, 0], [W, H, D]])

        if cam_pos_world is not None:
            nodes.append(cam_pos_world)

        if len(nodes) > 0:
            center = np.mean(nodes, axis=0)
        else:
            return None, None

        return canvas, {'center': center, 'rot_params': {'yaw': np.radians(yaw), 'pitch': np.radians(pitch)},
                        'offset': (W_IMG // 2, H_IMG // 2)}



    @staticmethod
    def _draw_3d_persons(canvas, persons, ctx, zoom, pixel_points, world_points, view_options, cam_pos_2d,
                         custom_rectangles=None):
        """Zeichnet die Personen als 3D-Säulen oder Skelett, abhängig von den verfügbaren Daten und Einstellungen."""
        show_skel = view_options.get("show_skeleton_3d", True)

        for p in persons:
            base_3d = p['pos']
            color = RenderColors.PERSON_STAND
            skel_3d = {}

            if show_skel and len(pixel_points) >= 4 and len(world_points) >= 4:
                skel_3d = GeometryMath.lift_skeleton_to_3d(p.get('keypoints', []), base_3d, pixel_points, world_points,
                                                           custom_rectangles)

            if skel_3d:
                label_pos = Renderer3D._draw_person_skeleton(canvas, skel_3d, color, ctx, zoom, cam_pos_2d)
            else:
                label_pos = Renderer3D._draw_person_pillar(canvas, base_3d, p.get('height', 180), color, ctx, zoom,
                                                           cam_pos_2d)

            name_label = p.get('status', f"ID:{p['id']}")
            if label_pos is not None:
                CalibrationRenderer.draw_hud_text(canvas, f"{name_label} | {p.get('height', 0):.0f}cm",
                                                  (label_pos[0] + 15, label_pos[1] - 15), color, 0.8, 1)

    @staticmethod
    def _draw_person_skeleton(canvas, skel_3d, color, ctx, zoom, cam_pos_2d):
        """Zeichnet das 3D-Skelett der Person. Zusätzlich wird eine vertikale Linie zum Boden gezeichnet, um die genaue Position zu verdeutlichen."""
        projected_joints = {
            j_id: GeometryMath.project_3d_to_2d(pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom) for
            j_id, pos in skel_3d.items()}

        Renderer3D._draw_sightlines(canvas, projected_joints.values(), cam_pos_2d)

        for pt_2d in projected_joints.values():
            cv2.circle(canvas, pt_2d, 4, color, -1, cv2.LINE_AA)

        for s, e in CalibrationRenderer.SKELETON_LINKS:
            if s in projected_joints and e in projected_joints:
                cv2.line(canvas, projected_joints[s], projected_joints[e], RenderColors.GRAY_LIGHT, 2, cv2.LINE_AA)

        if 11 in skel_3d and 12 in skel_3d:
            hip_center = (skel_3d[11] + skel_3d[12]) / 2
            p_hip = GeometryMath.project_3d_to_2d(hip_center, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            ground_pos = np.array([hip_center[0], 0.0, hip_center[2]])
            p_foot = GeometryMath.project_3d_to_2d(ground_pos, ctx['center'], ctx['rot_params'], ctx['offset'], zoom)

            # 1. Vertikale gestrichelte Linie nach unten (Gravitation / Ortung)
            CalibrationRenderer.draw_dashed_line(canvas, p_hip, p_foot, RenderColors.GRAY_DARK, 2, 6)

            # 2. Visuelle Ellipse auf dem Boden
            cv2.ellipse(canvas, p_foot, (20, 8), 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, p_foot, 2, RenderColors.WHITE, -1, cv2.LINE_AA)

        head_id = 0 if 0 in skel_3d else (5 if 5 in skel_3d else -1)
        return projected_joints[head_id] if head_id != -1 else (
            list(projected_joints.values())[0] if projected_joints else None)
