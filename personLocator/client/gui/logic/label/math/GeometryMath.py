import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging


class GeometryMath:
    camera_matrix = None
    dist_coeffs = None

    _last_log_state = ""
    _last_error_state = ""

    _last_debug_fingerprints = {}
    _pose_cache = {}

    @staticmethod
    def scale_camera_matrix(camera_matrix: np.ndarray, calib_res: list, current_res: list | tuple) -> np.ndarray:
        if camera_matrix is None or len(camera_matrix) == 0:
            return camera_matrix

        if current_res is None:
            current_res = (1920, 1080)

        mtx = np.array(camera_matrix, dtype=np.float64)
        real_calib_w = mtx[0, 2] * 2.0
        real_calib_h = mtx[1, 2] * 2.0

        if real_calib_w < 100:
            real_calib_w, real_calib_h = calib_res[0], calib_res[1]

        if abs(real_calib_w - current_res[0]) < 10 and abs(real_calib_h - current_res[1]) < 10:
            return mtx

        scale_x = current_res[0] / max(1.0, real_calib_w)
        scale_y = current_res[1] / max(1.0, real_calib_h)

        mtx[0, 0] *= scale_x
        mtx[1, 1] *= scale_y
        mtx[0, 2] *= scale_x
        mtx[1, 2] *= scale_y

        return mtx

    @staticmethod
    def project_3d_to_2d(p3d: np.ndarray, center: np.ndarray,
                         rotation_params: dict, offset: Tuple[int, int],
                         scale_factor: float = 1.0) -> Tuple[int, int]:
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

        if img_size is None:
            img_size = (1920, 1080)

        matrix_id = str(
            np.array(camera_matrix_override).flatten()[0]) if camera_matrix_override is not None else "default"

        # Hash über den gesamten Inhalt der Vierecke (alle Pixelkoordinaten),
        # damit CAMERA_1 einen anderen Schlüssel hat als CAMERA_2.
        rect_state = str(custom_rectangles) if custom_rectangles else "none"
        cache_key = f"{img_size}-{matrix_id}-{hash(rect_state)}"

        if cache_key in GeometryMath._pose_cache:
            return GeometryMath._pose_cache[cache_key]

        obj_pts = []
        img_pts = []
        log_messages = []
        used_rect_details = []

        main_valid_count = 0
        if pixel_points and world_points:
            for px_pt, w_pt in zip(pixel_points, world_points):
                if px_pt is None or w_pt is None: continue
                px = px_pt.get("px", 0) if isinstance(px_pt, dict) else px_pt[0]
                py = px_pt.get("py", 0) if isinstance(px_pt, dict) else px_pt[1]
                if px != 0 and py != 0:
                    main_valid_count += 1

        if main_valid_count == 0:
            log_messages.append("INFO: Keine Haupt-Raumpunkte (HUL, HUR...) geladen (Nur für Optik relevant).")
        else:
            log_messages.append(f"INFO: {main_valid_count} Haupt-Raumpunkte für Optik geladen (PnP ignoriert sie).")

        def safe_float(val):
            try:
                if val is None: return 0.0
                return float(str(val).replace(',', '.'))
            except ValueError:
                return 0.0

        custom_valid_count = 0
        if custom_rectangles:
            for rect in custom_rectangles:
                if not rect.get("is_active", True): continue
                if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("is_main_calib"): continue

                corners = rect.get("corners", [])
                if any(c.get("px") is None or c.get("py") is None for c in corners):
                    log_messages.append(f"WARNUNG: Viereck '{rect.get('display_id')}' ignoriert (Fehlende Pixel).")
                    continue

                xs = [safe_float(c.get("x")) for c in corners]
                ys = [safe_float(c.get("y")) for c in corners]
                zs = [safe_float(c.get("z")) for c in corners]
                pxs = [safe_float(c.get("px")) for c in corners]
                pys = [safe_float(c.get("py")) for c in corners]

                if (max(xs) - min(xs)) == 0 and (max(ys) - min(ys)) == 0 and (max(zs) - min(zs)) == 0:
                    continue
                if (max(pxs) - min(pxs)) == 0 and (max(pys) - min(pys)) == 0:
                    continue

                name = rect.get("display_id", rect.get("internal_id", "Unbekannt"))
                min_x, max_x = min(xs), max(xs)
                min_z, max_z = min(zs), max(zs)
                used_rect_details.append(
                    f"'{name}' [X: {min_x:.0f} bis {max_x:.0f} cm | Z: {min_z:.0f} bis {max_z:.0f} cm]")

                for c in corners:
                    obj_pts.append([safe_float(c.get("x")), safe_float(c.get("y")), safe_float(c.get("z"))])
                    img_pts.append([safe_float(c.get("px")), safe_float(c.get("py"))])
                    custom_valid_count += 1

        if custom_valid_count > 0:
            log_messages.append(f"INFO: {custom_valid_count} Custom-Viereck-Punkte für PnP geladen.")

        active_matrix = camera_matrix_override if camera_matrix_override is not None else GeometryMath.camera_matrix
        active_dist = dist_coeffs_override if dist_coeffs_override is not None else GeometryMath.dist_coeffs

        K = GeometryMath.scale_camera_matrix(active_matrix, [1920, 1080], img_size)
        if K is None:
            w, h = img_size
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(active_dist, dtype=np.float32) if active_dist is not None else np.zeros((4, 1),
                                                                                                       dtype=np.float32)

        success, rvec, tvec = False, None, None

        if len(obj_pts) >= 4:
            obj_pts_arr = np.array(obj_pts, dtype=np.float32)
            img_pts_arr = np.array(img_pts, dtype=np.float32)

            if not np.all(img_pts_arr == img_pts_arr[0]):
                try:
                    success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs,
                                                       flags=cv2.SOLVEPNP_SQPNP)
                    if not success:
                        success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
                except cv2.error:
                    try:
                        success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
                    except cv2.error:
                        success = False

        status_str = "ERFOLG" if success else "FEHLSCHLAG"
        debug_fingerprint = f"{status_str}-{img_size}-{len(obj_pts)}"

        if GeometryMath._last_debug_fingerprints.get(cache_key) != debug_fingerprint:
            GeometryMath._last_debug_fingerprints[cache_key] = debug_fingerprint

            print("\n" + "--- 📐 PNP KALIBRIERUNG DIAGNOSE ---")
            for msg in log_messages: print(msg)
            if len(obj_pts) > 0:
                print(f"-> Summe PnP-verwertbarer 3D-Punkte (NUR Vierecke!): {len(obj_pts)}")
                print(f"-> Beispiel Projektion: Pixel {img_pts[0]} ---> Raum {obj_pts[0]}")
            print("-------------------------------------")

            print("\n" + "=" * 65)
            print(f"📸 3D-KALIBRIERUNG: BERECHNUNGS-PROTOKOLL [{status_str}]")
            print("--- 1. VERWENDETE FAKTOREN (INPUTS) ---")
            print(f"   ➤ Auflösung:      {img_size}")
            print(f"   ➤ PnP-Punkte:     {len(obj_pts)}")

            print(f"   ➤ Genutzte Vierecke:")
            if used_rect_details:
                for detail in used_rect_details:
                    print(f"      ▫️ {detail}")
            else:
                print("      ▫️ KEINE")

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            print(f"   ➤ Kamera-Matrix:  Brennweite(fx:{fx:.1f}, fy:{fy:.1f}), Zentrum(cx:{cx:.1f}, cy:{cy:.1f})")
            print(f"   ➤ Lens Coeffs:    {dist_coeffs.flatten().tolist()}")

            print("\n--- 2. BERECHNETES ERGEBNIS (OUTPUT) ---")
            if success:
                R, _ = cv2.Rodrigues(rvec)
                pos = -np.dot(R.T, tvec).flatten()
                print(f"   🎯 3D-Position:   X: {pos[0]:.1f} | Y: {pos[1]:.1f} | Z: {pos[2]:.1f} cm")
            else:
                print(f"   ❌ Keine Pose berechnet. (Grund: Zu wenige Punkte oder solvePnP fehlgeschlagen)")
            print("=" * 65 + "\n")

        if success:
            GeometryMath._pose_cache[cache_key] = (rvec, tvec, K, dist_coeffs)
            return rvec, tvec, K, dist_coeffs

        GeometryMath._pose_cache[cache_key] = None
        return None

    @staticmethod
    def project_2d_to_3d(px: int, py: int, pixel_points: list, world_points: list,
                         custom_rectangles: list = None, target_y: float = 0.0,
                         img_size=(1920, 1080), camera_matrix_override=None, dist_coeffs_override=None) -> Optional[
        np.ndarray]:
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles, img_size,
                                            camera_matrix_override, dist_coeffs_override)
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
        if not keypoints or anchor_pos_3d is None: return {}
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles, img_size,
                                            camera_matrix_override, dist_coeffs_override)
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
                               img_size=(1920, 1080), camera_matrix_override=None, dist_coeffs_override=None) -> \
            Optional[np.ndarray]:
        center_x = int((bbox[0] + bbox[2]) / 2)
        bottom_y = int(bbox[3])
        return GeometryMath.project_2d_to_3d(center_x, bottom_y, pixel_points, world_points, custom_rectangles,
                                             target_y=0.0, img_size=img_size,
                                             camera_matrix_override=camera_matrix_override,
                                             dist_coeffs_override=dist_coeffs_override)

    @staticmethod
    def calculate_angle(p_center: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        vec_a, vec_b = p1 - p_center, p2 - p_center
        norm_prod = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm_prod == 0: return 0.0
        cos_theta = np.clip(np.dot(vec_a, vec_b) / norm_prod, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))