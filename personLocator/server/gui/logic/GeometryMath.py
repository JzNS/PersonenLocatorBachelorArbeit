import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging


class GeometryMath:
    """
    Volumetrischer Mathematik-Kern für echtes 3D-Tracking.
    Jetzt Multi-Kamera-fähig durch dynamische Matrix-Übergabe!
    """
    camera_matrix = None
    dist_coeffs = None
    _last_log_state = ""
    _last_error_state = ""

    @staticmethod
    def project_3d_to_2d(p3d: np.ndarray, center: np.ndarray,
                         rotation_params: dict, offset: Tuple[int, int],
                         scale_factor: float = 1.0) -> Tuple[int, int]:
        """Projektion eines 3D-Punkts auf die 2D-Ebene unter Berücksichtigung von Rotation und Skalierung."""
        ry, rp = rotation_params['yaw'], rotation_params['pitch']
        cy, sy, cp, sp = np.cos(ry), np.sin(ry), np.cos(rp), np.sin(rp)
        p_rel = p3d - center
        x_w, y_w, z_w = p_rel[0], p_rel[1] * -1, p_rel[2]
        y_rot = y_w * cp - z_w * sp
        z_rot = y_w * sp + z_w * cp
        x_f = x_w * cy + z_rot * sy
        return int((x_f * scale_factor) + offset[0]), int((y_rot * scale_factor) + offset[1])

    @staticmethod
    def get_camera_pose(pixel_points: List[Tuple[int, int]],
                        world_points: List[np.ndarray],
                        custom_rectangles: list = None,
                        img_size: Tuple[int, int] = (1920, 1080),
                        camera_matrix_override=None,
                        dist_coeffs_override=None) -> Optional[Tuple]:
        """Berechnet die Kamera-Pose (Rotation + Translation) basierend auf 2D-3D Korrespondenzen und optionalen Rechtecken."""
        obj_pts = []
        img_pts = []
        log_messages = []

        def safe_float(val):
            """Versucht, einen Wert sicher in Float umzuwandeln, auch wenn er als String mit Komma vorliegt."""
            try:
                if val is None: return 0.0
                return float(str(val).replace(',', '.'))
            except ValueError:
                return 0.0

        if custom_rectangles:
            for rect in custom_rectangles:
                if not rect.get("is_active", True): continue
                if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("is_main_calib"): continue
                corners = rect.get("corners", [])

                if any(c.get("px") is None or c.get("py") is None for c in corners): continue
                xs = [safe_float(c.get("x")) for c in corners]
                pxs = [safe_float(c.get("px")) for c in corners]
                pys = [safe_float(c.get("py")) for c in corners]

                if (max(pxs) - min(pxs)) == 0 and (max(pys) - min(pys)) == 0: continue

                for c in corners:
                    obj_pts.append([safe_float(c.get("x")), safe_float(c.get("y")), safe_float(c.get("z"))])
                    img_pts.append([safe_float(c.get("px")), safe_float(c.get("py"))])

        if len(obj_pts) < 4: return None

        obj_pts_arr = np.array(obj_pts, dtype=np.float32)
        img_pts_arr = np.array(img_pts, dtype=np.float32)
        if np.all(img_pts_arr == img_pts_arr[0]): return None

        if camera_matrix_override is not None and dist_coeffs_override is not None:
            K = np.array(camera_matrix_override, dtype=np.float32)
            dist_coeffs = np.array(dist_coeffs_override, dtype=np.float32)
        elif GeometryMath.camera_matrix is not None and GeometryMath.dist_coeffs is not None:
            K = np.array(GeometryMath.camera_matrix, dtype=np.float32)
            dist_coeffs = np.array(GeometryMath.dist_coeffs, dtype=np.float32)
        else:
            w, h = img_size
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if not success:
                success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success: return rvec, tvec, K, dist_coeffs
        except cv2.error:
            pass
        return None

    @staticmethod
    def project_2d_to_3d(px: int, py: int, pixel_points: list, world_points: list,
                         custom_rectangles: list = None, target_y: float = 0.0,
                         img_size=(1920, 1080), camera_matrix_override=None, dist_coeffs_override=None) -> Optional[np.ndarray]:
        """Projektion eines 2D-Pixels zurück in 3D, basierend auf der Kamera-Pose und einem Ziel-Y-Wert (z.B. Bodenhöhe)."""
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles, img_size, camera_matrix_override, dist_coeffs_override)
        if pose is None: return None

        rvec, tvec, K, dist_coeffs = pose
        R, _ = cv2.Rodrigues(rvec)
        R_inv = np.linalg.inv(R)
        C = -np.dot(R_inv, tvec).flatten()

        pt_2d = np.array([[[float(px), float(py)]]], dtype=np.float32)
        if np.any(dist_coeffs):
            pt_undist = cv2.undistortPoints(pt_2d, K, dist_coeffs)
            ray_cam = np.array([pt_undist[0][0][0], pt_undist[0][0][1], 1.0])
        else:
            K_inv = np.linalg.inv(K)
            ray_cam = np.dot(K_inv, np.array([px, py, 1.0]))

        ray_world = np.dot(R_inv, ray_cam)
        if abs(ray_world[1]) < 1e-6: return None
        s = (target_y - C[1]) / ray_world[1]
        if s < 0: return None
        P = C + s * ray_world
        return np.array([P[0], P[1], P[2]], dtype=np.float32)

    @staticmethod
    def lift_skeleton_to_3d(keypoints: list, anchor_pos_3d: np.ndarray,
                            pixel_points: list, world_points: list,
                            custom_rectangles: list = None,
                            img_size=(1920, 1080), camera_matrix_override=None, dist_coeffs_override=None) -> dict:
        """Hebt 2D-Keypoints in die 3D-Welt, basierend auf der Kamera-Pose und einem Ankerpunkt für die Höhe (z.B. Boden)."""
        if not keypoints or anchor_pos_3d is None: return {}
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles, img_size, camera_matrix_override, dist_coeffs_override)
        if pose is None: return {}

        rvec, tvec, K, dist_coeffs = pose
        R, _ = cv2.Rodrigues(rvec)
        R_inv = np.linalg.inv(R)
        C = -np.dot(R_inv, tvec).flatten()

        target_z = anchor_pos_3d[2]
        skeleton_3d = {}
        for k in keypoints:
            if k['c'] < 0.5: continue
            pt_2d = np.array([[[float(k['x']), float(k['y'])]]], dtype=np.float32)
            if np.any(dist_coeffs):
                pt_undist = cv2.undistortPoints(pt_2d, K, dist_coeffs)
                ray_cam = np.array([pt_undist[0][0][0], pt_undist[0][0][1], 1.0])
            else:
                K_inv = np.linalg.inv(K)
                ray_cam = np.dot(K_inv, np.array([k['x'], k['y'], 1.0]))

            ray_world = np.dot(R_inv, ray_cam)
            if abs(ray_world[2]) < 1e-6: continue
            s = (target_z - C[2]) / ray_world[2]
            if s < 0: continue
            P = C + s * ray_world
            if P[1] > 260 or P[1] < -20: continue
            skeleton_3d[k['id']] = np.array([P[0], P[1], P[2]], dtype=np.float32)
        return skeleton_3d

    @staticmethod
    def smart_project_position(keypoints: list, bbox: list, pixel_points: list, world_points: list,
                               person_height_cm: float = 180.0, custom_rectangles: list = None,
                               img_size=(1920, 1080), camera_matrix_override=None, dist_coeffs_override=None) -> Optional[np.ndarray]:
        """Projektion einer Person in die 3D-Welt, basierend auf der Kamera-Pose und der Annahme, dass die Person auf dem Boden steht (target_y=0)."""
        center_x = int((bbox[0] + bbox[2]) / 2)
        bottom_y = int(bbox[3])
        return GeometryMath.project_2d_to_3d(center_x, bottom_y, pixel_points, world_points, custom_rectangles,
                                             target_y=0.0, img_size=img_size, camera_matrix_override=camera_matrix_override, dist_coeffs_override=dist_coeffs_override)

    @staticmethod
    def calculate_angle(p_center: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """Berechnet den Winkel (in Grad) zwischen den Vektoren p_center->p1 und p_center->p2."""
        vec_a, vec_b = p1 - p_center, p2 - p_center
        norm_prod = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm_prod == 0: return 0.0
        cos_theta = np.clip(np.dot(vec_a, vec_b) / norm_prod, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))