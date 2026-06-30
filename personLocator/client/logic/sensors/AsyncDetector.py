import logging
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from client.gui.logic.CalibrationToolbox import CalibrationToolbox
from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.logic.sensors.PersonDetector import PersonDetector


class AsyncDetector(QThread):
    results_ready: pyqtSignal = pyqtSignal(list, list, dict)
    learn_data_ready: pyqtSignal = pyqtSignal(int, dict)

    def __init__(self, camera_name: str, detector: PersonDetector, toolbox: CalibrationToolbox) -> None:
        super().__init__()

        self.camera_name: str = camera_name
        self.detector: PersonDetector = detector
        self.toolbox: CalibrationToolbox = toolbox
        self._is_running: bool = True

        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_available: bool = False

        self.mutex: QMutex = QMutex()
        self.condition: QWaitCondition = QWaitCondition()
        self._cached_dead_mask: Optional[np.ndarray] = None
        self._cached_mirror_mask: Optional[np.ndarray] = None
        self._last_zone_hash: Optional[int] = None
        self._reset_requested: bool = False
        self.current_ai_time_ms: float = 33.3  # Startwert (entspricht ~30 FPS)
        self.time_mutex: QMutex = QMutex()
        self.roi_frame_counter: int = 0
        self.ROI_INTERVAL: int = 10
        self.MAX_ROI_PERSONS: int = 3
        self.current_roi_box: Optional[list[int]] = None  # globales [x1, y1, x2, y2]
        self.last_global_detections: list[dict[str, Any]] = []
        self.ROI_PADDING: int = 120

        self.processed_frames: int = 0
        self.last_fps_time: float = time.time()
        self._frame_tick: int = 0
        self._should_extract_colors: bool = False

    def get_processing_time(self) -> float:
        """Gibt die geglättete Berechnungszeit der KI thread-sicher zurück."""
        self.time_mutex.lock()
        val: float = self.current_ai_time_ms
        self.time_mutex.unlock()
        return val

    def update_toolbox(self, new_toolbox: CalibrationToolbox) -> None:
        """
        Thread-sicheres Update der Toolbox-Referenz (z.B. nach DB-Reload).
        Verhindert "Stale References" zwischen GUI und AI-Thread.
        """
        self.mutex.lock()
        self.toolbox = new_toolbox
        logging.info("AsyncDetector: Neue Toolbox-Referenz erfolgreich übernommen!")
        self.mutex.unlock()

    def _get_zone_masks(self, frame_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Erstellt oder liefert gecachte Masken für die Zonen-Filterung."""
        h: int = frame_shape[0]
        w: int = frame_shape[1]

        general_dead_zones: list[Any]
        mirror_zones: list[Any]
        general_dead_zones, mirror_zones = self.toolbox.zone_manager.get_zones()

        current_hash: int = hash(str(general_dead_zones) + str(mirror_zones))

        if self._last_zone_hash == current_hash and self._cached_dead_mask is not None and self._cached_mirror_mask is not None:
            return self._cached_dead_mask, self._cached_mirror_mask

        self._cached_dead_mask = np.zeros((h, w), dtype=np.uint8)
        self._cached_mirror_mask = np.zeros((h, w), dtype=np.uint8)

        parsed_dead: list[np.ndarray] = self._prep_polygon_arrays(general_dead_zones)
        parsed_mirror: list[np.ndarray] = self._prep_polygon_arrays(mirror_zones)

        if parsed_dead:
            cv2.fillPoly(self._cached_dead_mask, parsed_dead, 255)
        if parsed_mirror:
            cv2.fillPoly(self._cached_mirror_mask, parsed_mirror, 255)

        self._last_zone_hash = current_hash
        return self._cached_dead_mask, self._cached_mirror_mask

    def _match_ids_centroid(self, old_detections: list[dict[str, Any]], new_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Ordnet IDs in ROI-Frames über Distanzmessung zu.
        Inklusive dynamischem Anti-Jitter-Filter gegen KI-Halluzinationen bei Verdeckung!
        """
        used_ids: Set[int] = set()

        for new_d in new_detections:
            nx: float = (new_d["bbox"][0] + new_d["bbox"][2]) * 0.5
            ny: float = (new_d["bbox"][1] + new_d["bbox"][3]) * 0.5

            best_id: int = -1
            min_dist: float = 250.0
            best_old_d: Optional[dict[str, Any]] = None

            for old_d in old_detections:
                old_id: int = int(old_d.get("id", -1))

                if old_id in used_ids or old_id == -1:
                    continue

                ox: float = (old_d["bbox"][0] + old_d["bbox"][2]) * 0.5
                oy: float = (old_d["bbox"][1] + old_d["bbox"][3]) * 0.5
                dist: float = math.hypot(nx - ox, ny - oy)

                if dist < min_dist:
                    min_dist = dist
                    best_id = old_id
                    best_old_d = old_d

            if best_id != -1 and best_old_d is not None:
                new_d["id"] = best_id
                used_ids.add(best_id)

                ob: list[int] = best_old_d["bbox"]
                nb: list[int] = new_d["bbox"]

                old_h: int = ob[3] - ob[1]
                new_h: int = nb[3] - nb[1]

                alpha: float = 0.6

                # Bei abrupten Höhensprüngen (>15%) sehr stark dämpfen,
                # um Sprünge auf falsche Personen oder Halluzinationen zu glätten.
                if old_h > 0 and abs(new_h - old_h) / old_h > 0.15:
                    alpha = 0.1

                inv_alpha: float = 1.0 - alpha
                new_d["bbox"] = [
                    int(round((alpha * nb[0]) + (inv_alpha * ob[0]))),
                    int(round((alpha * nb[1]) + (inv_alpha * ob[1]))),
                    int(round((alpha * nb[2]) + (inv_alpha * ob[2]))),
                    int(round((alpha * nb[3]) + (inv_alpha * ob[3])))
                ]

            else:
                all_known_ids: list[int] = [int(d.get("id", 0)) for d in old_detections] + list(used_ids)
                new_id: int = (max(all_known_ids) + 1) if all_known_ids else 1
                new_d["id"] = new_id
                used_ids.add(new_id)

        return new_detections

    def update_frame(self, frame: np.ndarray, t1: Optional[float] = None) -> None:
        if frame is None:
            return
        self.mutex.lock()
        self._latest_frame = frame
        self._latest_t1 = t1
        self._new_frame_available = True
        self.condition.wakeAll()
        self.mutex.unlock()

    def reset_tracking(self) -> None:
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

    def _perform_safe_reset(self) -> None:
        """Überlässt YOLO den sicheren Reset, statt den Speicher gewaltsam zu löschen."""
        try:
            if hasattr(self, 'detector') and self.detector is not None:
                self.detector.last_resolution = None

            logging.info("AsyncDetector: YOLO-Tracker sicher zurückgesetzt.")
        except Exception as e:
            logging.error(f"Fehler beim Safe-Reset: {e}")

    def run(self) -> None:
        while self._is_running:
            self.mutex.lock()
            if not self._new_frame_available:
                self.condition.wait(self.mutex)

            frame_to_process: Optional[np.ndarray] = self._latest_frame
            current_t1: Optional[float] = self._latest_t1
            self._frame_tick += 1
            self._should_extract_colors = (self._frame_tick % 3 == 0)
            should_reset: bool = self._reset_requested

            self._new_frame_available = False
            self._reset_requested = False
            self.mutex.unlock()

            if should_reset:
                self._perform_safe_reset()
                self.last_global_detections = []
                self.current_roi_box = None
                self.roi_frame_counter = 0
                continue

            if frame_to_process is not None:
                start_ai_time: float = time.time()
                opts: dict[str, Any] = self.toolbox.view_options
                need_thumbs: bool = bool(opts.get("render_3d_enabled", True) and not opts.get("performance_mode", False))
                try:
                    yolo_frame: np.ndarray = frame_to_process
                    h_frame: int
                    w_frame: int
                    h_frame, w_frame = yolo_frame.shape[:2]

                    force_full_scan: bool = False

                    dynamic_padding: int = self.ROI_PADDING
                    dynamic_margin: int = 60

                    if self.current_roi_box is None:
                        force_full_scan = True
                    elif len(self.last_global_detections) > self.MAX_ROI_PERSONS:
                        force_full_scan = True
                    elif self.roi_frame_counter >= 30:
                        force_full_scan = True
                    else:
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

                    detections: list[dict[str, Any]]
                    t_inf_start = time.time()
                    if force_full_scan:
                        self.roi_frame_counter = 0
                        detections = self.detector.detect_persons(yolo_frame, use_tracking=True,
                                                                  generate_thumbnails=need_thumbs)

                        if self.last_global_detections:
                            detections = self._match_ids_centroid(self.last_global_detections, detections)

                        self.last_global_detections = detections

                        num_det: int = len(detections)
                        if 0 < num_det <= self.MAX_ROI_PERSONS:
                            min_x: int = min(d["bbox"][0] for d in detections)
                            min_y: int = min(d["bbox"][1] for d in detections)
                            max_x: int = max(d["bbox"][2] for d in detections)
                            max_y: int = max(d["bbox"][3] for d in detections)

                            self.current_roi_box = [
                                int(max(0, min_x - dynamic_padding)),
                                int(max(0, min_y - dynamic_padding)),
                                int(min(w_frame, max_x + dynamic_padding)),
                                int(min(h_frame, max_y + dynamic_padding))
                            ]
                        else:
                            self.current_roi_box = None
                    else:
                        self.roi_frame_counter += 1
                        rx1, ry1, rx2, ry2 = self.current_roi_box
                        roi_crop: np.ndarray = yolo_frame[ry1:ry2, rx1:rx2]

                        detections = self.detector.detect_persons(
                            roi_crop, offset_x=rx1, offset_y=ry1, use_tracking=False, generate_thumbnails=need_thumbs
                        )
                        detections = self._match_ids_centroid(self.last_global_detections, detections)
                        self.last_global_detections = detections
                    
                    inference_time_ms = (time.time() - t_inf_start) * 1000.0

                    h_full: int
                    w_full: int
                    h_full, w_full = frame_to_process.shape[:2]
                    dead_mask, mirror_mask = self._get_zone_masks(frame_to_process.shape)

                    gui_detections: list[dict[str, Any]] = []

                    for det in detections:
                        det["inference_time_ms"] = inference_time_ms
                        if current_t1:
                            det["t1"] = current_t1

                        bbox: list[int] = det.get("bbox", [0, 0, 0, 0])
                        if len(bbox) >= 4:
                            mid_x: int = int(max(0, min(w_full - 1, (bbox[0] + bbox[2]) * 0.5)))
                            feet_y: int = int(max(0, min(h_full - 1, bbox[3])))
                            center_y: int = int(max(0, min(h_full - 1, (bbox[1] + bbox[3]) * 0.5)))
                            head_y: int = int(max(0, min(h_full - 1, bbox[1] + ((bbox[3] - bbox[1]) * 0.1))))

                            if mirror_mask[head_y, mid_x] == 255 or \
                                    mirror_mask[center_y, mid_x] == 255 or \
                                    mirror_mask[feet_y, mid_x] == 255:
                                continue

                            if dead_mask[center_y, mid_x] == 255:
                                continue

                        gui_detections.append(det)

                    person_results: list[dict[str, Any]] = self._process_3d_logic(gui_detections, frame_to_process)
                    
                    results_meta = {
                        "t1": current_t1 or 0.0,
                        "inference_time_ms": inference_time_ms
                    }
                    self.results_ready.emit(gui_detections, person_results, results_meta)

                    self.processed_frames += 1
                    now: float = time.time()
                    time_diff: float = now - self.last_fps_time

                    if time_diff >= 60.0:
                        fps: float = self.processed_frames / time_diff
                        logging.info(
                            f"🧠 [KI Stats] Verarbeitungs-Rate: {fps:.1f} FPS ({self.processed_frames} Bilder in {time_diff:.1f}s analysiert)")
                        self.processed_frames = 0
                        self.last_fps_time = now

                except Exception as e:
                    logging.error(f"Async Detector Fehler im Loop: {e}")
                finally:
                    p_time: float = (time.time() - start_ai_time) * 1000.0

                    self.time_mutex.lock()
                    # EMA-Filter: 80% alte Zeit, 20% neue Zeit
                    self.current_ai_time_ms = (self.current_ai_time_ms * 0.8) + (p_time * 0.2)
                    self.time_mutex.unlock()

    def _prep_polygon_arrays(self, zones: list[Any]) -> list[np.ndarray]:
        """
        Wandelt Zonen-Datenstrukturen einmalig pro Frame in schnelle Numpy-Arrays um.
        Verhindert CPU-Overhead in Schleifen.
        """
        parsed: list[np.ndarray] = []
        for polygon in zones:
            if isinstance(polygon, dict):
                polygon = polygon.get("points", polygon.get("polygon", []))

            if not polygon or not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue

            try:
                pts: list[list[float]] = []
                for p in polygon:
                    if hasattr(p, 'x') and hasattr(p, 'y'):
                        px: float = float(p.x() if callable(p.x) else p.x)
                        py: float = float(p.y() if callable(p.y) else p.y)
                        pts.append([px, py])
                    elif isinstance(p, dict):
                        pts.append([float(p.get('x', 0.0)), float(p.get('y', 0.0))])
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append([float(p[0]), float(p[1])])

                if len(pts) >= 3:
                    parsed.append(np.array(pts, dtype=np.int32))
            except Exception:
                pass
        return parsed

    def _process_3d_logic(self, detections: list[dict[str, Any]], frame: np.ndarray) -> list[dict[str, Any]]:
        """Hier findet die gesamte 3D-Logik statt, inklusive Höhenberechnung, Raycasting und Body-Metrics."""
        rendering_active: bool = bool(self.toolbox.view_options.get("render_3d_enabled", True))
        performance_mode: bool = bool(self.toolbox.view_options.get("performance_mode", False))
        pixel_points_snap: list[Any] = list(
            self.toolbox.pixel_points) if not performance_mode and self.toolbox.pixel_points else []
        world_points_snap: list[np.ndarray] = list(
            self.toolbox.world_points_3d) if not performance_mode and self.toolbox.world_points_3d else []
        is_calibrated: bool = not performance_mode and len(self.toolbox.custom_rectangles) > 0

        occluded_ids: Set[int] = set()
        num_dets: int = len(detections)
        for i in range(num_dets):
            for j in range(i + 1, num_dets):
                b1: list[int] = detections[i].get("bbox", [0, 0, 0, 0])
                b2: list[int] = detections[j].get("bbox", [0, 0, 0, 0])
                if (b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]):
                    occluded_ids.add(int(detections[i].get("id", 0)))
                    occluded_ids.add(int(detections[j].get("id", 0)))

        results: list[dict[str, Any]] = []

        for det in detections:
            bbox: list[int] = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            p_id: int = int(det.get("id", 0))
            kps_rich: list[dict[str, Any]] = det.get("keypoints", [])
            bbox_conf: float = float(det.get("confidence", 0.0))
            thumb: Optional[np.ndarray] = det.get("thumbnail") if rendering_active else None
            is_occluded: bool = p_id in occluded_ids

            kp_map: dict[int, dict[str, Any]] = {int(kp['id']): kp for kp in kps_rich if isinstance(kp, dict) and 'id' in kp}

            if performance_mode:
                live_colors: dict[int, str] = {}
                body_metrics: dict[str, Any] = {}
                if not is_occluded and frame is not None and kps_rich and self._should_extract_colors:
                    valid_kps_for_color: dict[int, dict[str, Any]] = {
                        kp_id: kp_map[kp_id]
                        for kp_id in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                        if kp_id in kp_map and float(kp_map[kp_id]['c']) > 0.90
                    }
                    live_colors = self._extract_colors_batched_numpy(frame, valid_kps_for_color)

                if hasattr(self.toolbox, 'person_manager'):
                    pm: Any = self.toolbox.person_manager
                    color_data: dict[str, Any] = pm.update_colors(p_id, live_colors, "General")
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
                    "distance": 0.0, "confidence": bbox_conf, "tilt_angle": 0.0, "tilt_direction": "-",
                    "t1": det.get("t1", 0.0), "inference_time_ms": det.get("inference_time_ms", 0.0)
                })
                continue

            pos_3d: np.ndarray = np.array([0, 0, 0], dtype=np.float32)
            res_data: dict[str, Any] = {"height": 0.0, "distance": 0.0, "confidence": 0.0, "is_skeleton": False}
            stable_h: float = 0.0
            tilt_angle: float = 0.0
            tilt_direction: str = "Gerade"
            body_metrics = {}

            k11, k12, k23, k24 = kp_map.get(11), kp_map.get(12), kp_map.get(23), kp_map.get(24)
            if all([k11, k12, k23, k24]):
                mid_sh_x: float = (float(k11['x']) + float(k12['x'])) * 0.5
                mid_sh_y: float = (float(k11['y']) + float(k12['y'])) * 0.5
                mid_hp_x: float = (float(k23['x']) + float(k24['x'])) * 0.5
                mid_hp_y: float = (float(k23['y']) + float(k24['y'])) * 0.5
                angle: float = math.atan2(mid_sh_y - mid_hp_y, mid_sh_x - mid_hp_x)
                tilt_angle = round(math.degrees(angle) - 90.0, 1)
                tilt_direction = "Links" if tilt_angle > 5.0 else "Rechts" if tilt_angle < -5.0 else "Gerade"

            if is_calibrated and kps_rich:
                try:
                    ref_h: float = 180.0
                    if hasattr(self.toolbox, 'person_manager'):
                        ref_h = float(self.toolbox.person_manager.get_ref_height(p_id))

                    p3d: Optional[np.ndarray] = GeometryMath.smart_project_position(
                        kps_rich, bbox, pixel_points_snap,
                        world_points_snap, ref_h,
                        self.toolbox.custom_rectangles,
                        img_size=self.toolbox.current_resolution
                    )

                    if p3d is not None:
                        pos_3d = p3d
                        res_data = self.toolbox.calculate_height_and_confidence(bbox, pos_3d, kps_rich)
                        body_metrics = self.toolbox.analyze_body_metrics(kps_rich, pos_3d, frame)

                        if hasattr(self.toolbox, 'person_manager'):
                            pm_3d: Any = self.toolbox.person_manager

                            pm_3d.update_height_measurement(p_id, res_data["height"])
                            status_text_3d: str = str(det.get("status", ""))
                            stable_h = float(pm_3d.update_height_ema(p_id, res_data["height"], status_text_3d))

                            body_metrics["total_height"] = stable_h
                            body_metrics["body_tilt"] = tilt_angle
                            body_metrics["tilt_direction"] = tilt_direction

                            live_width: float = float(body_metrics.get("shoulder_width", 0.0))
                            angle_orient: float = float(body_metrics.get("orientation_angle", 0.0))

                            if abs(angle_orient) < 25 or abs(angle_orient) > 155:
                                pm_3d.update_width_measurement(p_id, live_width)

                            learned_width: float = float(pm_3d.local_bests["width"].get(p_id, 0.0))
                            body_metrics["learned_width"] = int(learned_width)

                            orientation_3d: str = str(body_metrics.get("orientation", "Unbekannt"))

                            if is_occluded or not self._should_extract_colors:
                                live_colors_3d: dict[int, str] = {}
                            else:
                                valid_kps_for_color_3d: dict[int, dict[str, Any]] = {
                                    kp_id: kp_map[kp_id]
                                    for kp_id in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                                    if kp_id in kp_map and float(kp_map[kp_id]['c']) > 0.90
                                }
                                live_colors_3d = self._extract_colors_batched_numpy(frame, valid_kps_for_color_3d)

                                body_metrics["joint_colors"] = live_colors_3d

                            color_data_3d: dict[str, Any] = pm_3d.update_colors(p_id, live_colors_3d, orientation_3d)

                            body_metrics["color_profiles"] = color_data_3d["detailed"]
                            body_metrics["stable_colors"] = color_data_3d["display"]
                        else:
                            body_metrics["total_height"] = res_data["height"]

                except Exception as e:
                    logging.error(f"3D-Logik Fehler für ID {p_id}: {e}")

            results.append({
                "id": p_id, "bbox": bbox, "bbox_confidence": bbox_conf, "pos": pos_3d,
                "height": res_data["height"], "stable_height": stable_h, "metrics": body_metrics,
                "distance": res_data["distance"], "confidence": res_data["confidence"],
                "status": f"Dist: {int(res_data['distance'])}cm", "thumbnail": thumb,
                "keypoints": kps_rich, "tilt_angle": tilt_angle, "tilt_direction": tilt_direction,
                "t1": det.get("t1", 0.0), "inference_time_ms": det.get("inference_time_ms", 0.0)
            })

        return results

    def _extract_colors_batched_numpy(self, frame: np.ndarray, keypoints_to_check: Dict[int, Dict[str, Any]], window_size: int = 7) -> Dict[int, str]:
        """
        Extrahiert die Median-Farbe aller validen Keypoints in einem einzigen,
        vektorisierten NumPy-Aufruf. Verhindert CPU-Overhead durch Python-Schleifen,
        ohne fehleranfällige CUDA-Kompilierung zu benötigen.
        """
        if not keypoints_to_check:
            return {}

        h: int
        w: int
        h, w = frame.shape[:2]
        half: int = window_size // 2

        valid_ids: List[int] = []
        rois: List[np.ndarray] = []

        for kp_id, kp in keypoints_to_check.items():
            x: int = int(kp['x'])
            y: int = int(kp['y'])

            x1: int = max(0, x - half)
            y1: int = max(0, y - half)
            x2: int = min(w, x + half + 1)
            y2: int = min(h, y + half + 1)

            # Nur vollständig sichtbare ROIs für sauberes Batching zulassen
            if x2 - x1 == window_size and y2 - y1 == window_size:
                rois.append(frame[y1:y2, x1:x2])
                valid_ids.append(kp_id)

        if not rois:
            return {}

        # Shape: (N, window, window, 3)
        batch_rois: np.ndarray = np.stack(rois)

        medians_cpu: np.ndarray = np.median(batch_rois, axis=(1, 2)).astype(np.uint8)

        results: Dict[int, str] = {}
        for i, kp_id in enumerate(valid_ids):
            b, g, r = medians_cpu[i]
            results[kp_id] = f"#{r:02x}{g:02x}{b:02x}"

        return results

    def stop(self) -> None:
        self._is_running = False
        self.condition.wakeAll()
        self.wait()