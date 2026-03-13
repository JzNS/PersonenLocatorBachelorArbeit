from typing import Tuple, Dict, Optional

import cv2
import numpy as np
from client.gui.logic.label.math.GeometryMath import GeometryMath


class CalibrationMath:
    """
    Volumetrische Analyse-Engine.
    Berechnet Metriken (Längen, Winkel, Höhen) ausschließlich auf Basis
    von echten 3D-Vektoren (Raycasting) anstatt von 2D-Pixel-Näherungen.
    """

    @staticmethod
    def generate_geometric_analysis(world_points: list, labels: list, connections: list, angles: list) -> dict:
        """Berechnet räumliche Metriken wie echte 3D-Distanzen
         und Winkel zwischen definierten Referenzpunkten zur Visualisierung im UI-Panel."""
        if not world_points:
            return {}

        analysis = {}
        for i, label in enumerate(labels):
            if i >= len(world_points):
                continue

            p_curr = world_points[i]
            entries = []

            for s, e in connections:
                nb = e if s == i else (s if e == i else -1)
                if nb != -1 and nb < len(world_points):
                    dist = np.linalg.norm(p_curr - world_points[nb])
                    entries.append({"type": "Länge", "name": f"zu {labels[nb]}", "value": f"{dist:.1f}"})

            for c, a, b in angles:
                if c == i and max(a, b) < len(world_points):
                    deg = GeometryMath.calculate_angle(p_curr, world_points[a], world_points[b])
                    entries.append({"type": "Winkel", "name": f"zw. {labels[a]} & {labels[b]}", "value": f"{deg:.1f}°"})

            analysis[label] = entries

        return analysis

    @staticmethod
    def calculate_confidence_and_distance(pos_3d: np.ndarray) -> Tuple[float, float]:
        """Ermittelt die Distanz einer Person zur Kamera in Zentimetern
        und berechnet einen Konfidenzwert, der bei geringem Abstand (unter 1,5 m)
        aufgrund möglicher Verdeckungen abnimmt."""
        dist_to_cam = float(np.linalg.norm(pos_3d))
        confidence = 1.0

        if dist_to_cam < 150:
            confidence = np.clip((dist_to_cam - 40) / 110, 0.1, 1.0)

        return round(dist_to_cam, 1), round(confidence, 2)

    @staticmethod
    def evaluate_raycast_precision(px: int, py: int, ideal_3d: np.ndarray,
                                   pixel_points: list, world_points: list,
                                   custom_rectangles: list) -> dict:
        """
        Bewertet die Genauigkeit eines Raycast-Punktes durch die kombinierte
         Berechnung von physikalischem Welt-Fehler (cm), 2D-Reprojektionsfehler
         (Pixel) und räumlicher Sensitivität.
        """
        real_3d = GeometryMath.project_2d_to_3d(px, py, pixel_points, world_points,
                                                custom_rectangles, target_y=ideal_3d[1])

        if real_3d is None:
            return {"status": "RAY FAIL", "error_cm": 999.0, "pixel_drift": 999.0, "sensitivity": 0.0}

        error_cm = np.linalg.norm(real_3d - ideal_3d)

        pose = GeometryMath.get_camera_pose(pixel_points, world_points, custom_rectangles)
        pixel_drift = 0.0
        if pose:
            rvec, tvec, K, dist = pose
            pts_2d, _ = cv2.projectPoints(np.array([ideal_3d], dtype=np.float32), rvec, tvec, K, dist)
            pixel_drift = np.linalg.norm(pts_2d[0][0] - np.array([px, py]))

        neighbor_3d = GeometryMath.project_2d_to_3d(px + 1, py, pixel_points, world_points,
                                                    custom_rectangles, target_y=ideal_3d[1])
        sensitivity = np.linalg.norm(neighbor_3d - real_3d) if neighbor_3d is not None else 0.0

        return {
            "status": "OK",
            "error_cm": error_cm,
            "pixel_drift": pixel_drift,
            "sensitivity": sensitivity,
            "real_3d": real_3d
        }

    @staticmethod
    def analyze_body_metrics(keypoints: list, pos_3d: np.ndarray,
                             pixel_points: list, world_points: list,
                             correction_factor: float = 1.0,
                             frame: np.ndarray = None, custom_rectangles: list = None) -> dict:
        """Überführt 2D-Skelett-Keypoints in den 3D-Raum, um anatomische Proportionen
        (wie Arm- und Beinlänge), die Körperausrichtung, Gelenkfarben und 3D-Vektoren
        zu extrahieren."""
        metrics = {
            "orientation": "Unbekannt",
            "orientation_angle": 0.0,
            "orientation_confidence": 0.0,
            "arm_len": 0, "leg_len": 0, "shoulder_width": 0,
            "torso_color": None, "joint_colors": {}
        }

        if not keypoints or not pixel_points or pos_3d is None:
            return metrics

        skeleton_3d = GeometryMath.lift_skeleton_to_3d(keypoints, pos_3d, pixel_points, world_points, custom_rectangles)

        if not skeleton_3d:
            return metrics

        def get_kp2d(idx):
            return next((k for k in keypoints if k['id'] == idx), {'x': 0, 'y': 0, 'c': 0})

        # --- 2. Farben Sicherheitsprüfung ---
        if frame is not None:
            relevant_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            h_img, w_img = frame.shape[:2]

            def get_col(px, py):
                ix, iy = int(px), int(py)
                if 0 <= ix < w_img and 0 <= iy < h_img:
                    b, g, r = frame[iy, ix]
                    return f"#{r:02x}{g:02x}{b:02x}"
                return None

            for rid in relevant_ids:
                kp = get_kp2d(rid)
                if kp['c'] > 0.85:
                    metrics["joint_colors"][rid] = get_col(kp['x'], kp['y'])

        # --- 3.  3D-PROPORTIONEN  ---
        def dist_3d(id1, id2):
            if id1 in skeleton_3d and id2 in skeleton_3d:
                # tilt_factor komplett entfernt
                return np.linalg.norm(skeleton_3d[id1] - skeleton_3d[id2]) * correction_factor
            return 0.0

        arm_len = max(dist_3d(5, 7) + dist_3d(7, 9), dist_3d(6, 8) + dist_3d(8, 10))
        leg_len = max(dist_3d(11, 13) + dist_3d(13, 15), dist_3d(12, 14) + dist_3d(14, 16))

        shoulder_width_3d = dist_3d(5, 6)

        metrics["arm_len"] = int(arm_len)
        metrics["leg_len"] = int(leg_len)
        metrics["shoulder_width"] = int(shoulder_width_3d)

        # --- 4. ORIENTIERUNG & ROTATION (Heuristik über Projektions-Stauchung) ---
        has_ls, has_rs = 5 in skeleton_3d, 6 in skeleton_3d
        has_lh, has_rh = 11 in skeleton_3d, 12 in skeleton_3d
        nose_c = get_kp2d(0)['c']

        angle = 0.0
        conf = 0.0
        orientation = "Unbekannt"

        if has_ls and has_rs:
            if nose_c > 0.4:
                if 5.0 < shoulder_width_3d < 32.0:
                    mid_shoulder_x = (get_kp2d(5)['x'] + get_kp2d(6)['x']) / 2
                    if get_kp2d(0)['x'] < mid_shoulder_x:
                        orientation = "Schräg (Links)"
                        angle = -45.0
                    else:
                        orientation = "Schräg (Rechts)"
                        angle = 45.0
                    conf = 0.8
                else:
                    orientation = "Front View"
                    angle = 0.0
                    conf = 0.95
            else:
                orientation = "Back View"
                angle = 180.0
                conf = 0.9
        elif has_ls or has_lh:
            orientation = "Side (Links)"
            angle = -90.0
            conf = 0.85
        elif has_rs or has_rh:
            orientation = "Side (Rechts)"
            angle = 90.0
            conf = 0.85

        metrics["orientation"] = orientation
        metrics["orientation_angle"] = angle
        metrics["orientation_confidence"] = conf

        metrics["skeleton_3d"] = {}
        if skeleton_3d:
            for j_id, vec_3d in skeleton_3d.items():
                metrics["skeleton_3d"][j_id] = {
                    "x": round(float(vec_3d[0]), 1),
                    "y": round(float(vec_3d[1]), 1),
                    "z": round(float(vec_3d[2]), 1)
                }

        return metrics

    @staticmethod
    def calculate_person_height(pixel_bbox: list,
                                pos_3d: np.ndarray,
                                keypoints: Optional[list],
                                pixel_points: list,
                                world_points_3d: list,
                                custom_rectangles: list = None) -> Tuple[float, bool]:
        """
        Bestimmt die exakte Körpergröße in Zentimetern,
        vorzugsweise durch die Aufsummierung von 3D-Skelettsegmenten
        oder alternativ über das Raycasting der oberen Bounding-Box-Kante.
        """
        if len(pixel_points) < 4 or pos_3d is None:
            return 180.0, False

        used_skeleton = False
        final_height = 0.0

        if keypoints:
            skeleton_3d = GeometryMath.lift_skeleton_to_3d(keypoints, pos_3d, pixel_points, world_points_3d,
                                                           custom_rectangles)
            if skeleton_3d and len(skeleton_3d) > 5:
                final_height = CalibrationMath.sum_skeleton_segments_3d(skeleton_3d)

                if final_height < 50.0:
                    final_height = max([p[1] for p in skeleton_3d.values()])

                used_skeleton = True

        # Methode B: Bounding Box Raycasting
        if not used_skeleton and len(pixel_bbox) >= 4:
            center_x = (pixel_bbox[0] + pixel_bbox[2]) / 2.0
            top_y = pixel_bbox[1]

            fake_kps = [{'id': 99, 'x': center_x, 'y': top_y, 'c': 1.0}]
            bbox_3d = GeometryMath.lift_skeleton_to_3d(fake_kps, pos_3d, pixel_points, world_points_3d,
                                                       custom_rectangles)

            if 99 in bbox_3d:
                final_height = bbox_3d[99][1]
            else:
                final_height = 180.0

        # Plausibilitäts-Check
        final_height = np.clip(final_height, 50.0, 250.0)

        return float(final_height), used_skeleton

    @staticmethod
    def sum_skeleton_segments_3d(skel_3d: Dict[int, np.ndarray]) -> float:
        """Berechnet die kumulierte Länge der wichtigsten 3D-Knochensegmente
        einer Körperhälfte (Kopf bis Fuß), um die Körpergröße auch bei
        nicht-aufrechter Haltung präzise zu messen."""

        def segment_len(p1_id, p2_id, p3_id, p4_id, p5_id):
            pts = [skel_3d.get(i) for i in [p1_id, p2_id, p3_id, p4_id, p5_id]]
            if any(p is None for p in pts):
                return 0.0
            return (np.linalg.norm(pts[0] - pts[1]) +
                    np.linalg.norm(pts[1] - pts[2]) +
                    np.linalg.norm(pts[2] - pts[3]) +
                    np.linalg.norm(pts[3] - pts[4]))

        left_side = segment_len(0, 5, 11, 13, 15)
        right_side = segment_len(0, 6, 12, 14, 16)

        return max(left_side, right_side)

