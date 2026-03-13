import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import logging


class PersonDetector:
    """
    Performance-Optimiertes KI-Modul. (Ohne Pose-Classifier)
    """

    CLASS_PERSON = 0
    CONF_THRESHOLD = 0.4

    SKELETON_CONNECTIONS = np.array([
        [5, 7], [7, 9],  # Linker Arm
        [6, 8], [8, 10],  # Rechter Arm
        [11, 13], [13, 15],  # Linkes Bein
        [12, 14], [14, 16],  # Rechtes Bein
        [5, 6], [11, 12],  # Schulter & Hüfte
        [5, 11], [6, 12]  # Torso
    ])

    def __init__(self, model_variant: str = "client/config/yolov8n-pose_openvino_model/"):
        try:
            self.model = YOLO(model_variant, task="pose")
        except Exception as e:
            logging.warning(f"Standard-Modell wird geladen (Fehler: {e})")
            self.model = YOLO("client/config/yolov8n-pose.pt")

        self.last_resolution = None

    def detect_persons(self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0, use_tracking: bool = True) -> \
    List[Dict[str, Any]]:
        """Erkennt Personen im Bild und gibt eine Liste von Detections zurück.
        Jede Detection enthält BBox, Keypoints, ID, Status und Farbe für die Anzeige."""
        h_img, w_img = frame.shape[:2]
        current_res = (w_img, h_img)

        should_persist = True
        if use_tracking:
            if self.last_resolution is None or self.last_resolution != current_res:
                should_persist = False
                if self.last_resolution is not None:
                    logging.info(f"YOLO-Tracker Reset: Auflösung {self.last_resolution} -> {current_res}")
                self.last_resolution = current_res

        if use_tracking:
            results = self.model.track(frame, persist=should_persist, verbose=False)
        else:
            results = self.model.predict(frame, verbose=False)

        detections = []
        if not results or results[0].boxes is None:
            return []

        r = results[0]
        boxes_all = r.boxes.xyxy.cpu().numpy()
        confs_all = r.boxes.conf.cpu().numpy()

        if r.boxes.id is not None:
            ids_all = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids_all = np.zeros(len(boxes_all), dtype=int)

        has_kps = r.keypoints is not None and r.keypoints.xy is not None
        if has_kps:
            kps_xy_all = r.keypoints.xy.cpu().numpy()
            kps_conf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

        for i, box in enumerate(boxes_all):
            conf = confs_all[i]
            if conf <= self.CONF_THRESHOLD:
                continue

            local_x1 = max(0, min(w_img, int(box[0])))
            local_y1 = max(0, min(h_img, int(box[1])))
            local_x2 = max(0, min(w_img, int(box[2])))
            local_y2 = max(0, min(h_img, int(box[3])))

            if local_x2 <= local_x1 or local_y2 <= local_y1:
                continue

            global_bbox = [local_x1 + offset_x, local_y1 + offset_y, local_x2 + offset_x, local_y2 + offset_y]

            current_kps_local = kps_xy_all[i] if has_kps else None
            current_kps_conf = kps_conf_all[i] if has_kps and kps_conf_all is not None else None

            current_kps_global = None
            if current_kps_local is not None:
                current_kps_global = current_kps_local.copy()
                current_kps_global[:, 0] += offset_x
                current_kps_global[:, 1] += offset_y

            # --- Status & Farbe---
            status_display = f"Sicher: {int(conf * 100)}%"
            color = (0, 255, 0)

            rich_keypoints = []
            if current_kps_global is not None:
                rich_keypoints = [
                    {"id": k_idx, "x": float(kx), "y": float(ky),
                     "c": float(current_kps_conf[k_idx] if current_kps_conf is not None else 0)}
                    for k_idx, (kx, ky) in enumerate(current_kps_global)
                ]

            combined_thumbnail = None
            try:
                real_crop = frame[local_y1:local_y2, local_x1:local_x2]
                TARGET_H = 200
                crop_h, crop_w = real_crop.shape[:2]
                if crop_h > 0 and crop_w > 0:
                    scale = TARGET_H / crop_h
                    target_w = int(crop_w * scale)
                    real_rs = cv2.resize(real_crop, (target_w, TARGET_H), interpolation=cv2.INTER_LINEAR)
                    cv2.rectangle(real_rs, (0, 0), (target_w - 1, TARGET_H - 1), (0, 255, 0), 2)
                    skel_rs = np.zeros_like(real_rs)
                    if current_kps_local is not None:
                        kps_small = current_kps_local.copy()
                        kps_small[:, 0] = (kps_small[:, 0] - local_x1) * scale
                        kps_small[:, 1] = (kps_small[:, 1] - local_y1) * scale
                        self._draw_skeleton_fast(skel_rs, kps_small)
                        self._draw_skeleton_fast(real_rs, kps_small)
                    combined_thumbnail = np.hstack([skel_rs, real_rs])
            except Exception:
                pass

            detections.append({
                "id": int(ids_all[i]),
                "bbox": global_bbox,
                "keypoints": rich_keypoints,
                "thumbnail": combined_thumbnail,
                "status": status_display,
                "confidence": float(conf),
                "color": color
            })

        return detections

    def _draw_skeleton_fast(self, canvas: np.ndarray, kps: np.ndarray):
        """Zeichnet das Skelett auf das gegebene Canvas.
        Optimiert für Geschwindigkeit, indem nur gültige Linien und Punkte gezeichnet werden."""
        h, w = canvas.shape[:2]
        kps_int = kps.astype(int)
        for i in range(len(self.SKELETON_CONNECTIONS)):
            idx_a, idx_b = self.SKELETON_CONNECTIONS[i]
            if idx_a >= len(kps) or idx_b >= len(kps): continue
            pt_a, pt_b = kps_int[idx_a], kps_int[idx_b]
            if (0 <= pt_a[0] < w and 0 <= pt_a[1] < h and 0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
                cv2.line(canvas, pt_a, pt_b, (255, 255, 0), 2, cv2.LINE_AA)
        for pt in kps_int:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(canvas, pt, 3, (255, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], scale: float = 1.0) -> np.ndarray:
        """Zeichnet die Bounding-Boxen, IDs und Statusinformationen der Detections auf den Frame."""
        if not detections: return frame
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) < 4: continue
            try:
                x1, y1 = int(round(float(bbox[0]) * scale)), int(round(float(bbox[1]) * scale))
                x2, y2 = int(round(float(bbox[2]) * scale)), int(round(float(bbox[3]) * scale))

                color = det.get("color", (0, 255, 0))
                if not isinstance(color, tuple): color = tuple(int(c) for c in color)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"ID:{det.get('id', '?')} | {det.get('status', '').split(' ')[0]}"
                text_y = y1 - 10 if y1 > 20 else y1 + 20
                font_scale = 0.6 if scale > 0.5 else 0.4
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

                cv2.rectangle(frame, (int(x1), int(text_y - th - 5)), (int(x1 + tw), int(text_y + 5)), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (int(x1), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            except Exception as e:
                logging.error(f"Fehler beim Zeichnen der Bbox: {e}")
        return frame