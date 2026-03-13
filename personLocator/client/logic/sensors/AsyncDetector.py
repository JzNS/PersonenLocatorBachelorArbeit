import logging
import math
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import copy
from client.gui.logic.CalibrationToolbox import CalibrationToolbox
from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.logic.sensors.PersonDetector import PersonDetector


class AsyncDetector(QThread):
    results_ready = pyqtSignal(list, list)
    learn_data_ready = pyqtSignal(int, dict)

    def __init__(self, camera_name: str, detector: PersonDetector, toolbox: CalibrationToolbox):
        super().__init__()

        self.camera_name = camera_name
        self.detector = detector
        self.toolbox = toolbox
        self._is_running = True

        # Frame Puffer
        self._latest_frame = None
        self._new_frame_available = False

        # Thread-Safety & Synchronisation
        self.mutex = QMutex()
        self.condition = QWaitCondition()

        #Thread Sicherheit
        self._reset_requested = False

        #ROI Tracking
        self.roi_frame_counter = 0
        self.ROI_INTERVAL = 10
        self.MAX_ROI_PERSONS = 3
        self.current_roi_box = None  # Speichert das globale [x1, y1, x2, y2]
        self.last_global_detections = []
        self.ROI_PADDING = 120

    def update_toolbox(self, new_toolbox: CalibrationToolbox) -> None:
        """
        Thread-sicheres Update der Toolbox-Referenz (z.B. nach DB-Reload).
        Verhindert "Stale References" zwischen GUI und AI-Thread.
        """
        self.mutex.lock()
        self.toolbox = new_toolbox
        logging.info("AsyncDetector: Neue Toolbox-Referenz erfolgreich übernommen!")
        self.mutex.unlock()
    def _match_ids_centroid(self, old_detections: list, new_detections: list) -> list:
        """
        Ordnet IDs in ROI-Frames über Distanzmessung zu.
        Inklusive dynamischem Anti-Jitter-Filter gegen KI-Halluzinationen bei Verdeckung!
        """
        used_ids = set()

        for new_d in new_detections:
            nx = (new_d["bbox"][0] + new_d["bbox"][2]) / 2
            ny = (new_d["bbox"][1] + new_d["bbox"][3]) / 2

            best_id = -1
            min_dist = 250.0
            best_old_d = None

            for old_d in old_detections:
                old_id = old_d.get("id", -1)

                if old_id in used_ids or old_id == -1:
                    continue

                ox = (old_d["bbox"][0] + old_d["bbox"][2]) / 2
                oy = (old_d["bbox"][1] + old_d["bbox"][3]) / 2
                dist = math.hypot(nx - ox, ny - oy)

                if dist < min_dist:
                    min_dist = dist
                    best_id = old_id
                    best_old_d = old_d

            if best_id != -1 and best_old_d is not None:
                new_d["id"] = best_id
                used_ids.add(best_id)

                # ==========================================
                # ANTI-JITTER FILTER
                # ==========================================
                ob = best_old_d["bbox"]
                nb = new_d["bbox"]

                old_h = ob[3] - ob[1]
                new_h = nb[3] - nb[1]

                # Standard-Glättung: 60% Vertrauen in das neue Bild, 40% ins alte Bild
                alpha = 0.6

                # HALLUZINATIONS-SCHUTZ:

                if old_h > 0 and abs(new_h - old_h) / old_h > 0.15:
                    alpha = 0.1

                new_d["bbox"] = [
                    int(round((alpha * nb[0]) + ((1 - alpha) * ob[0]))),
                    int(round((alpha * nb[1]) + ((1 - alpha) * ob[1]))),
                    int(round((alpha * nb[2]) + ((1 - alpha) * ob[2]))),
                    int(round((alpha * nb[3]) + ((1 - alpha) * ob[3])))
                ]

            else:
                all_known_ids = [d.get("id", 0) for d in old_detections] + list(used_ids)
                new_id = (max(all_known_ids) + 1) if all_known_ids else 1
                new_d["id"] = new_id
                used_ids.add(new_id)

        return new_detections
    def update_frame(self, frame: np.ndarray):
        if frame is None: return
        self.mutex.lock()
        self._latest_frame = frame.copy()
        self._new_frame_available = True
        self.condition.wakeAll()
        self.mutex.unlock()

    def reset_tracking(self):
        """
        Signalisiert dem AI-Thread, dass er sich vor dem nächsten Frame resetten soll.
        Wird vom GUI-Thread aufgerufen.
        """
        self.mutex.lock()
        self._latest_frame = None
        self._new_frame_available = False
        self._reset_requested = True
        logging.info("AsyncDetector: Reset angefordert (Warte auf AI-Thread).")
        self.mutex.unlock()

    def _perform_safe_reset(self):
        """Überlässt YOLO den sicheren Reset, statt den Speicher gewaltsam zu löschen."""
        try:
            if hasattr(self, 'detector') and self.detector is not None:
                self.detector.last_resolution = None

            logging.info("AsyncDetector: YOLO-Tracker sicher zurückgesetzt.")
        except Exception as e:
            logging.error(f"Fehler beim Safe-Reset: {e}")

    def run(self):
        while self._is_running:
            self.mutex.lock()
            if not self._new_frame_available:
                self.condition.wait(self.mutex)

            # Daten übernehmen
            frame_to_process = self._latest_frame
            should_reset = self._reset_requested

            # Flags zurücksetzen
            self._new_frame_available = False
            self._reset_requested = False
            self.mutex.unlock()

            # 1. Sicherer Reset VOR der Verarbeitung (falls angefordert)
            if should_reset:
                self._perform_safe_reset()
                self.last_global_detections = []
                self.current_roi_box = None
                self.roi_frame_counter = 0
                continue

            if frame_to_process is not None:
                try:
                    # 2. Render Capacity Logik
                    cap_percent = self.toolbox.view_options.get("render_capacity", 100)
                    scale = max(0.1, cap_percent / 100.0)  # Schutz vor 0

                    if scale < 1.0:
                        h, w = frame_to_process.shape[:2]
                        # Bild verkleinern
                        yolo_frame = cv2.resize(frame_to_process, (int(w * scale), int(h * scale)),
                                                interpolation=cv2.INTER_LINEAR)
                    else:
                        yolo_frame = frame_to_process

                        # ==========================================
                        # 3. ADAPTIVE STICKY ROI DETEKTION
                        # ==========================================
                    force_full_scan = False

                    # Padding und Margin dynamisch an den Schieberegler anpassen
                    dynamic_padding = int(self.ROI_PADDING * scale)
                    dynamic_margin = int(60 * scale)

                    if self.current_roi_box is None:
                        force_full_scan = True
                    elif len(self.last_global_detections) > self.MAX_ROI_PERSONS:
                        force_full_scan = True
                    elif self.roi_frame_counter >= 30:
                        force_full_scan = True
                    else:
                        # B. Hysterese mit KORREKTEM Coordinate-Space
                        rx1, ry1, rx2, ry2 = self.current_roi_box

                        if rx2 <= rx1 or ry2 <= ry1:
                            force_full_scan = True
                        else:
                            for det in self.last_global_detections:
                                bx1, by1, bx2, by2 = det["bbox"]
                                if (bx1 < rx1 + dynamic_margin or
                                        by1 < ry1 + dynamic_margin or
                                        bx2 > rx2 - dynamic_margin or
                                        by2 > ry2 - dynamic_margin):
                                    force_full_scan = True
                                    break

                    h_frame, w_frame = yolo_frame.shape[:2]

                    if force_full_scan:
                        # --- FULL SCAN ---
                        self.roi_frame_counter = 0
                        detections = self.detector.detect_persons(yolo_frame, use_tracking=True)

                        # IDs sichern, da der Tracker-Reset neue IDs generiert hat
                        if self.last_global_detections:
                            detections = self._match_ids_centroid(self.last_global_detections, detections)

                        self.last_global_detections = detections

                        #"Sticky" ROI Bounding-Box berechnen und EINFRIEREN
                        num_det = len(detections)
                        if 0 < num_det <= self.MAX_ROI_PERSONS:
                            min_x = min(d["bbox"][0] for d in detections)
                            min_y = min(d["bbox"][1] for d in detections)
                            max_x = max(d["bbox"][2] for d in detections)
                            max_y = max(d["bbox"][3] for d in detections)

                            self.current_roi_box = [
                                int(max(0, min_x - dynamic_padding)),
                                int(max(0, min_y - dynamic_padding)),
                                int(min(w_frame, max_x + dynamic_padding)),
                                int(min(h_frame, max_y + dynamic_padding))
                            ]
                        else:
                            self.current_roi_box = None
                    else:
                        # --- ROI SCAN ---
                        self.roi_frame_counter += 1
                        rx1, ry1, rx2, ry2 = self.current_roi_box
                        roi_crop = yolo_frame[ry1:ry2, rx1:rx2]

                        detections = self.detector.detect_persons(
                            roi_crop, offset_x=rx1, offset_y=ry1, use_tracking=False
                        )
                        detections = self._match_ids_centroid(self.last_global_detections, detections)
                        self.last_global_detections = detections
                        # 4. Hochskalieren (BBox und Keypoints)
                        gui_detections = copy.deepcopy(detections)
                        if scale < 1.0 and gui_detections:
                            for det in gui_detections:
                                if "bbox" in det and isinstance(det["bbox"], (list, np.ndarray)) and len(
                                        det["bbox"]) >= 4:
                                    det["bbox"] = [int(c / scale) for c in det["bbox"]]

                                if "keypoints" in det and isinstance(det["keypoints"], list):
                                    for kp in det["keypoints"]:
                                        if isinstance(kp, dict) and "x" in kp and "y" in kp:
                                            kp["x"] /= scale
                                            kp["y"] /= scale

                        # ==========================================
                        # ZONEN-FILTERUNG AUF GUI-DATEN
                        # ==========================================
                        general_dead_zones, mirror_zones = ([], [])
                        if hasattr(self.toolbox, "zone_manager"):
                            general_dead_zones, mirror_zones = self.toolbox.zone_manager.get_zones()

                        filtered_gui_detections = []
                        for det in gui_detections:
                            bbox = det.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                feet_x = (bbox[0] + bbox[2]) / 2
                                feet_y = bbox[3]
                                center_x, center_y = feet_x, (bbox[1] + bbox[3]) / 2

                                if any(self._is_point_in_polygon(feet_x, feet_y, p) for p in mirror_zones):
                                    continue
                                if any(self._is_point_in_polygon(center_x, center_y, p) for p in general_dead_zones):
                                    continue

                            filtered_gui_detections.append(det)

                        gui_detections = filtered_gui_detections
                        person_results = self._process_3d_logic(gui_detections, frame_to_process)
                        self.results_ready.emit(gui_detections, person_results)

                except Exception as e:
                    logging.error(f"Async Detector Fehler im Loop: {e}")

    def _is_point_in_polygon(self, x: float, y: float, polygon) -> bool:
        """
        Prüft, ob ein Punkt im Polygon liegt.
        Kugelsicher gegen Dictionaries, Listen und PyQt-Objekte!
        """
        if isinstance(polygon, dict):
            polygon = polygon.get("points", polygon.get("polygon", []))

        if not polygon or not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            return False

        try:
            pts = []
            for p in polygon:
                if hasattr(p, 'x') and hasattr(p, 'y'):
                    px = p.x() if callable(p.x) else p.x
                    py = p.y() if callable(p.y) else p.y
                    pts.append([float(px), float(py)])
                elif isinstance(p, dict):
                    pts.append([float(p.get('x', 0)), float(p.get('y', 0))])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([float(p[0]), float(p[1])])

            if len(pts) < 3:
                return False

            poly_array = np.array(pts, dtype=np.int32)
            return cv2.pointPolygonTest(poly_array, (float(x), float(y)), False) >= 0

        except Exception as e:
            logging.error(f"Kritischer Fehler bei der Polygon-Prüfung: {e}")
            return False
    def _process_3d_logic(self, detections: list, frame: np.ndarray) -> list:
        """Hier findet die gesamte 3D-Logik statt, inklusive Höhenberechnung, Raycasting und Body-Metrics."""
        rendering_active = self.toolbox.view_options.get("render_3d_enabled", True)
        performance_mode = self.toolbox.view_options.get("performance_mode", False)
        pixel_points_snap = list(
            self.toolbox.pixel_points) if not performance_mode and self.toolbox.pixel_points else []
        world_points_snap = list(
            self.toolbox.world_points_3d) if not performance_mode and self.toolbox.world_points_3d else []
        is_calibrated = not performance_mode and len(self.toolbox.custom_rectangles) > 0

        # 2D-Verdeckungs-Prüfung
        occluded_ids = set()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                b1 = detections[i].get("bbox", [0, 0, 0, 0])
                b2 = detections[j].get("bbox", [0, 0, 0, 0])
                if (b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]):
                    occluded_ids.add(detections[i].get("id", 0))
                    occluded_ids.add(detections[j].get("id", 0))

        results = []

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4: continue

            feet_x = (bbox[0] + bbox[2]) / 2
            feet_y = bbox[3]
            center_x, center_y = feet_x, (bbox[1] + bbox[3]) / 2


            p_id = det.get("id", 0)
            kps_rich = det.get("keypoints", [])
            bbox_conf = det.get("confidence", 0.0)
            thumb = det.get("thumbnail") if rendering_active else None
            is_occluded = p_id in occluded_ids

            # ==========================================
            # A. PERFORMANCE MODUS (NUR 2D + FARBEN)
            # ==========================================
            if performance_mode:
                live_colors = {}

                if not is_occluded and frame is not None and kps_rich:
                    h_img, w_img = frame.shape[:2]
                    relevant_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    for kp in kps_rich:
                        if kp['id'] in relevant_ids and kp['c'] > 0.90:
                            color_hex = self._extract_robust_color(frame, int(kp['x']), int(kp['y']))
                            if color_hex:
                                live_colors[kp['id']] = color_hex

                body_metrics = {}

                if hasattr(self.toolbox, 'person_manager'):
                    pm = self.toolbox.person_manager
                    color_data = pm.update_colors(p_id, live_colors, "General")
                    if not color_data.get("display") and live_colors:
                        pm.force_color_sync(p_id)
                        color_data = pm.update_colors(p_id, live_colors, "General")

                    body_metrics["color_profiles"] = color_data.get("detailed", {})
                    body_metrics["stable_colors"] = color_data.get("display", {})

                body_metrics["joint_colors"] = live_colors

                results.append({
                    "id": p_id, "bbox": bbox, "bbox_confidence": bbox_conf,
                    "keypoints": kps_rich, "status": "Performance Mode",
                    "thumbnail": thumb, "pos": np.array([0, 0, 0], dtype=np.float32),
                    "height": 0.0, "stable_height": 0.0, "metrics": body_metrics,
                    "distance": 0.0, "confidence": bbox_conf, "tilt_angle": 0.0, "tilt_direction": "-"
                })
                continue

            # ==========================================
            # B. FULL 3D MODUS (METRIKEN & RAYCASTING)
            # ==========================================
            pos_3d = np.array([0, 0, 0], dtype=np.float32)
            res_data = {"height": 0.0, "distance": 0.0, "confidence": 0.0, "is_skeleton": False}
            stable_h, tilt_angle, tilt_direction = 0.0, 0.0, "Gerade"
            body_metrics = {}

            def get_k(idx):
                if not kps_rich: return None
                return next((k for k in kps_rich if isinstance(k, dict) and k.get('id') == idx), None)

            k11, k12, k23, k24 = get_k(11), get_k(12), get_k(23), get_k(24)
            if all([k11, k12, k23, k24]):
                mid_sh_x, mid_sh_y = (k11['x'] + k12['x']) / 2, (k11['y'] + k12['y']) / 2
                mid_hp_x, mid_hp_y = (k23['x'] + k24['x']) / 2, (k23['y'] + k24['y']) / 2
                angle = math.atan2(mid_sh_y - mid_hp_y, mid_sh_x - mid_hp_x)
                tilt_angle = round(math.degrees(angle) - 90.0, 1)
                tilt_direction = "Links" if tilt_angle > 5.0 else "Rechts" if tilt_angle < -5.0 else "Gerade"

            if is_calibrated and kps_rich:
                try:
                    ref_h = 180.0
                    if hasattr(self.toolbox, 'person_manager'):
                        ref_h = self.toolbox.person_manager.get_ref_height(p_id)

                    p3d = GeometryMath.smart_project_position(
                        kps_rich, bbox, pixel_points_snap,
                        world_points_snap, ref_h,
                        self.toolbox.custom_rectangles
                    )

                    if p3d is not None:
                        pos_3d = p3d
                        res_data = self.toolbox.calculate_height_and_confidence(bbox, pos_3d, kps_rich)
                        body_metrics = self.toolbox.analyze_body_metrics(kps_rich, pos_3d, frame)

                        if hasattr(self.toolbox, 'person_manager'):
                            pm = self.toolbox.person_manager

                            stable_h = pm.update_height_measurement(p_id, res_data["height"])
                            body_metrics["total_height"] = stable_h
                            body_metrics["body_tilt"] = tilt_angle
                            body_metrics["tilt_direction"] = tilt_direction

                            live_width = body_metrics.get("shoulder_width", 0)
                            angle_orient = body_metrics.get("orientation_angle", 0)

                            if abs(angle_orient) < 25 or abs(angle_orient) > 155:
                                pm.update_width_measurement(p_id, live_width)

                            learned_width = pm.local_bests["width"].get(p_id, 0.0)
                            body_metrics["learned_width"] = int(learned_width)

                            orientation = body_metrics.get("orientation", "Unbekannt")

                            if is_occluded:
                                live_colors = {}
                            else:
                                high_conf_colors = {}
                                relevant_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                                for kp in kps_rich:
                                    if kp['c'] > 0.90 and kp['id'] in relevant_ids:
                                        color_hex = self._extract_robust_color(frame, int(kp['x']), int(kp['y']))
                                        if color_hex:
                                            high_conf_colors[kp['id']] = color_hex
                                live_colors = high_conf_colors
                                body_metrics["joint_colors"] = live_colors

                            color_data = pm.update_colors(p_id, live_colors, orientation)

                            body_metrics["color_profiles"] = color_data["detailed"]
                            body_metrics["stable_colors"] = color_data["display"]
                        else:
                            body_metrics["total_height"] = res_data["height"]

                except Exception as e:
                    logging.error(f"3D-Logik Fehler für ID {p_id}: {e}")

            results.append({
                "id": p_id, "bbox": bbox, "bbox_confidence": bbox_conf, "pos": pos_3d,
                "height": res_data["height"], "stable_height": stable_h, "metrics": body_metrics,
                "distance": res_data["distance"], "confidence": res_data["confidence"],
                "status": f"Dist: {int(res_data['distance'])}cm", "thumbnail": thumb,
                "keypoints": kps_rich, "tilt_angle": tilt_angle, "tilt_direction": tilt_direction
            })

        return results
    def _extract_robust_color(self, frame: np.ndarray, x: int, y: int, window_size: int = 7) -> str:
        """
        Sammelt Pixel in einem Bereich um das Gelenk und gibt den Median-Farbwert zurück.
        Ignoriert Ausreißer wie Schatten, helle Reflexionen oder einzelne Hintergrund-Pixel!
        """
        h, w = frame.shape[:2]
        half = window_size // 2

        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half + 1), min(h, y + half + 1)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None

        b = int(np.median(roi[:, :, 0]))
        g = int(np.median(roi[:, :, 1]))
        r = int(np.median(roi[:, :, 2]))

        return f"#{r:02x}{g:02x}{b:02x}"
    def stop(self):
        self._is_running = False
        self.condition.wakeAll()
        self.wait()