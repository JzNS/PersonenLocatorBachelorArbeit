import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging


class GeometryMath:

    camera_matrix = None
    dist_coeffs = None

    _last_log_state = ""
    _last_error_state = ""

    @staticmethod
    def project_3d_to_2d(p3d: np.ndarray, center: np.ndarray,
                         rotation_params: dict, offset: Tuple[int, int],
                         scale_factor: float = 1.0) -> Tuple[int, int]:
        """Transformiert einen 3D-Punkt basierend auf
        gegebenen Rotationsparametern (Yaw, Pitch) sowie Skalierungs- und Offset-Werten
        in eine relative 2D-Pixelkoordinate."""
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
                        img_size: Tuple[int, int] = (1920, 1080)) -> Optional[Tuple]:
        """Berechnet die Kamerapose (Rotations- und Translationsvektoren)
        über das PnP-Verfahren (Perspective-n-Point) anhand definierter
        3D-2D-Korrespondenzen und bietet ein intelligentes,
        absturzsicheres Diagnose-Logging."""
        obj_pts = []
        img_pts = []
        log_messages = []

        # --- 1. Haupt-Raum-Kalibrierung -> Ist nur für die Optik ( um zu sehen, was die Kamera im Raum sieht) ---
        main_valid_count = 0
        if pixel_points and world_points:
            for px_pt, w_pt in zip(pixel_points, world_points):
                if px_pt is None or w_pt is None: continue

                if isinstance(px_pt, dict):
                    px, py = px_pt.get("px", 0), px_pt.get("py", 0)
                else:
                    px, py = px_pt[0], px_pt[1]

                if px != 0 and py != 0:
                    main_valid_count += 1

        if main_valid_count == 0:
            log_messages.append("INFO: Keine Haupt-Raumpunkte (HUL, HUR...) geladen (Nur für Optik relevant).")
        else:
            log_messages.append(f"INFO: {main_valid_count} Haupt-Raumpunkte für Optik geladen (PnP ignoriert sie).")

        # --- 2. Custom Vierecke auswerten (PnP) ---
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
                if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("is_main_calib"):
                    continue
                corners = rect.get("corners", [])

                if any(c.get("px") is None or c.get("py") is None for c in corners):
                    log_messages.append(f"WARNUNG: Viereck '{rect.get('display_id')}' ignoriert (Fehlende Pixel).")
                    continue

                xs = [safe_float(c.get("x")) for c in corners]
                ys = [safe_float(c.get("y")) for c in corners]
                zs = [safe_float(c.get("z")) for c in corners]

                pxs = [safe_float(c.get("px")) for c in corners]
                pys = [safe_float(c.get("py")) for c in corners]

                # Überspringen, wenn es keine echten 3D-Maße hat
                if (max(xs) - min(xs)) == 0 and (max(ys) - min(ys)) == 0 and (max(zs) - min(zs)) == 0:
                    log_messages.append(
                        f"WARNUNG: Viereck '{rect.get('display_id')}' ignoriert (Keine 3D-Raummaße gesetzt).")
                    continue

                if (max(pxs) - min(pxs)) == 0 and (max(pys) - min(pys)) == 0:
                    log_messages.append(
                        f"WARNUNG: Viereck '{rect.get('display_id')}' ignoriert (Alle Pixel am selben Punkt).")
                    continue

                for c in corners:
                    obj_pts.append([safe_float(c.get("x")), safe_float(c.get("y")), safe_float(c.get("z"))])
                    img_pts.append([safe_float(c.get("px")), safe_float(c.get("py"))])
                    custom_valid_count += 1

        if custom_valid_count > 0:
            log_messages.append(f"INFO: {custom_valid_count} Custom-Viereck-Punkte für PnP geladen.")

        current_state = f"M:{main_valid_count}|C:{custom_valid_count}"
        if current_state != getattr(GeometryMath, '_last_log_state', ''):
            GeometryMath._last_log_state = current_state
            logging.info("--- 📐 PNP KALIBRIERUNG DIAGNOSE ---")
            for msg in log_messages:
                logging.info(msg)
            logging.info(f"-> Summe PnP-verwertbarer 3D-Punkte (NUR Vierecke!): {len(obj_pts)}")
            if len(obj_pts) > 0:
                logging.info(f"-> Beispiel Projektion: Pixel {img_pts[0]} ---> Raum {obj_pts[0]}")
            logging.info("-------------------------------------")

        # --- 3. PnP Berechnen ---
        if len(obj_pts) == 0:
            if getattr(GeometryMath, '_last_error_state', '') != "perf_mode":
                logging.info("INFO: PnP-Kalibrierung ist deaktiviert (Performance-Modus aktiv oder keine Daten).")
                GeometryMath._last_error_state = "perf_mode"
            return None

        if len(obj_pts) < 4:
            return None

        obj_pts_arr = np.array(obj_pts, dtype=np.float32)
        img_pts_arr = np.array(img_pts, dtype=np.float32)

        # Prüfen, ob alle Punkte identisch sind (verhindert den OpenCV Absturz)
        if np.all(img_pts_arr == img_pts_arr[0]):
            return None

        obj_pts_arr = np.array(obj_pts, dtype=np.float32)
        img_pts_arr = np.array(img_pts, dtype=np.float32)

        if GeometryMath.camera_matrix is not None and GeometryMath.dist_coeffs is not None:
            K = np.array(GeometryMath.camera_matrix, dtype=np.float32)
            dist_coeffs = np.array(GeometryMath.dist_coeffs, dtype=np.float32)
        else:
            w, h = img_size
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success = False
        rvec = None
        tvec = None

        try:
            success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if not success:
                success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as e:
            try:
                success, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, K, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                success = False

        if success:
            if getattr(GeometryMath, '_last_error_state', '') != "success":
                logging.info("✅ PNP ERFOLGREICH: Kameraposition exakt berechnet!")
                GeometryMath._last_error_state = "success"
            return rvec, tvec, K, dist_coeffs
        else:
            if getattr(GeometryMath, '_last_error_state', '') != "pnp_failed":
                logging.error("❌ PNP FEHLER: solvePnP gescheitert (Punkte liegen in einer Linie oder am selben Ort).")
                GeometryMath._last_error_state = "pnp_failed"
            return None

    @staticmethod
    def project_2d_to_3d(px: int, py: int, pixel_points: list, world_points: list,
                         custom_rectangles: list = None, target_y: float = 0.0) -> Optional[np.ndarray]:
        """Projiziert eine 2D-Pixelkoordinate mittels Raycasting durch
        die berechnete Kamerapose zurück in den 3D-Raum, indem der
        Sichtstrahl mit einer definierten Y-Ebene (Zielhöhe) geschnitten wird."""
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles)
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
    def calculate_angle(p_center: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """Berechnet den Winkel in Grad zwischen drei 3D-Punkten,
        ausgehend von einem zentralen Scheitelpunkt (p_center) zu zwei Endpunkten."""
        vec_a, vec_b = p1 - p_center, p2 - p_center
        norm_prod = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm_prod == 0: return 0.0
        cos_theta = np.clip(np.dot(vec_a, vec_b) / norm_prod, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    @staticmethod
    def lift_skeleton_to_3d(keypoints: list, anchor_pos_3d: np.ndarray,
                            pixel_points: list, world_points: list,
                            custom_rectangles: list = None) -> dict:
        """Überführt 2D-Skelett-Keypoints in den 3D-Raum,
        indem die virtuellen Sichtstrahlen der Kamera mit einer
        definierten Z-Tiefenebene (Ankerposition) geschnitten werden."""
        if not keypoints or anchor_pos_3d is None: return {}
        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles)
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
                               person_height_cm: float = 180.0, custom_rectangles: list = None) -> Optional[np.ndarray]:
        """Schätzt die physikalische 3D-Bodenposition einer Person,
        indem die untere Mitte ihrer 2D-Bounding-Box über Raycasting
        auf die Raum-Null-Ebene (Y=0) projiziert wird."""
        center_x = int((bbox[0] + bbox[2]) / 2)
        bottom_y = int(bbox[3])
        return GeometryMath.project_2d_to_3d(center_x, bottom_y, pixel_points, world_points, custom_rectangles,
                                             target_y=0.0)