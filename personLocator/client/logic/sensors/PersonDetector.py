import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

import openvino as ov


class PersonDetector:
    """
    HYBRID ULTRA-Performance-Modul.
    Scannt die Hardware beim Start:
    1. Prio: NVIDIA GPU via ONNX Runtime (CUDA)
    2. Prio: Intel CPU/iGPU via OpenVINO
    """

    CLASS_PERSON: int = 0
    CONF_THRESHOLD: float = 0.4
    INPUT_SIZE: tuple[int, int] = (640, 640)

    SKELETON_CONNECTIONS: np.ndarray = np.array([
        [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16],
        [5, 6], [11, 12], [5, 11], [6, 12]
    ])

    def __init__(self,
                 model_xml_path: Optional[Union[str, Path]] = None,
                 model_onnx_path: Optional[Union[str, Path]] = None) -> None:

        self.backend: str = "UNKNOWN"
        self.actual_device: str = "UNKNOWN"
        self.last_resolution: Optional[tuple[int, int]] = None

        CLIENT_DIR: Path = Path(__file__).resolve().parent.parent.parent

        if model_xml_path is None:
            self.model_xml_path: Path = CLIENT_DIR / "config" / "yolo26n-pose_openvino_model" / "yolo26n-pose.xml"
        else:
            self.model_xml_path = Path(model_xml_path)

        if model_onnx_path is None:
            self.model_onnx_path: Path = CLIENT_DIR / "config" / "graka" / "yolo26n-pose.onnx"
        else:
            self.model_onnx_path = Path(model_onnx_path)

        logging.info("Hardware-Scan für KI-Inference startet...")

        self.ort_session: Optional[Any] = None
        self.input_name: Optional[str] = None
        try:
            import onnxruntime as ort
            providers: list[str] = ort.get_available_providers()

            if 'CUDAExecutionProvider' in providers:
                if self.model_onnx_path.exists():
                    # str() Cast ist wichtig für die C++ Bindings von ONNX/OpenVINO
                    self.ort_session = ort.InferenceSession(str(self.model_onnx_path), providers=['CUDAExecutionProvider'])
                    self.input_name = str(self.ort_session.get_inputs()[0].name)

                    self.backend = "ONNX"
                    self.actual_device = "NVIDIA GPU (CUDA)"
                    logging.info(f"✅ NVIDIA Grafikkarte gefunden! Lade ONNX Runtime ({self.model_onnx_path.name})")
                    return
                else:
                    logging.warning(
                        f"CUDA verfügbar, aber ONNX-Modell ({self.model_onnx_path.name}) fehlt. Fallback auf OpenVINO...")
        except ImportError:
            logging.info("onnxruntime nicht installiert. Überspringe CUDA-Suche.")
        except Exception as e:
            logging.warning(f"Fehler beim Laden von ONNX/CUDA: {e}. Fallback auf OpenVINO...")

        self.core: Optional[ov.Core] = None
        self.model: Optional[ov.Model] = None
        self.compiled_model: Optional[ov.CompiledModel] = None
        self.input_layer: Optional[Any] = None
        self.output_layer: Optional[Any] = None

        try:
            if not self.model_xml_path.exists():
                raise FileNotFoundError(f"Modell-Datei nicht gefunden: {self.model_xml_path}")

            self.core = ov.Core()
            self.model = self.core.read_model(str(self.model_xml_path))

            # "AUTO" sucht sich selbst den besten Intel-Chip (CPU oder integrierte GPU)
            self.compiled_model = self.core.compile_model(self.model, "AUTO")

            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)

            exec_device: str
            try:
                exec_device = str(self.compiled_model.get_property("EXECUTION_DEVICE"))
            except RuntimeError:
                exec_device = "AUTO (CPU/iGPU)"

            self.backend = "OPENVINO"
            self.actual_device = f"Intel OpenVINO ({exec_device})"
            logging.info(f"✅ Fallback erfolgreich! Lade OpenVINO auf: {self.actual_device}")

        except Exception as e:
            logging.critical(f"Kritischer Fehler beim Laden des OpenVINO Modells: {e}")
            raise e

    def detect_persons(self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0, use_tracking: bool = True, generate_thumbnails: bool = False) -> list[dict[str, Any]]:
        """Bildaufbereitung, Hybrid-Inference und manuelle Tensor-Dekodierung."""

        if frame is None or frame.size == 0:
            return []

        h_img, w_img = frame.shape[:2]
        scale_x: float = w_img / self.INPUT_SIZE[0]
        scale_y: float = h_img / self.INPUT_SIZE[1]

        input_tensor: np.ndarray = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=self.INPUT_SIZE,
            swapRB=True,
            crop=False
        )

        res: np.ndarray
        try:
            if self.backend == "ONNX" and self.ort_session:
                raw_results: list[np.ndarray] = self.ort_session.run(None, {self.input_name: input_tensor})
                res = raw_results[0]
            elif self.compiled_model:
                res = self.compiled_model([input_tensor])[self.output_layer]
            else:
                return []
        except Exception as e:
            logging.error(f"Inference Fehler ({self.backend}): {e}")
            return []

        predictions: np.ndarray = res[0]
        mask: np.ndarray = (predictions[:, 4] > self.CONF_THRESHOLD) & (predictions[:, 5] == self.CLASS_PERSON)
        valid_predictions: np.ndarray = predictions[mask]

        detections: list[dict[str, Any]] = []
        person_counter: int = 1

        for p in valid_predictions:
            conf: float = float(p[4])

            # --- Bounding Box ---
            x1: int = max(0, min(w_img, int(p[0] * scale_x)))
            y1: int = max(0, min(h_img, int(p[1] * scale_y)))
            x2: int = max(0, min(w_img, int(p[2] * scale_x)))
            y2: int = max(0, min(h_img, int(p[3] * scale_y)))

            if x2 <= x1 or y2 <= y1:
                continue

            global_bbox: list[int] = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]

            # --- Keypoints ---
            raw_kps: np.ndarray = p[6:57].reshape(17, 3)
            rich_keypoints: list[dict[str, Any]] = []
            local_kps_for_thumbnail: list[list[float]] = []

            for k_idx, kp in enumerate(raw_kps):
                kp_x: float = float(kp[0] * scale_x)
                kp_y: float = float(kp[1] * scale_y)
                kp_conf: float = float(kp[2])

                local_kps_for_thumbnail.append([kp_x, kp_y, kp_conf])

                rich_keypoints.append({
                    "id": k_idx,
                    "x": kp_x + offset_x,
                    "y": kp_y + offset_y,
                    "c": kp_conf
                })

            local_kps_array: np.ndarray = np.array(local_kps_for_thumbnail)
            status_display: str = f"Sicher: {int(conf * 100)}%"
            color: tuple[int, int, int] = (0, 255, 0)
            combined_thumbnail: Optional[np.ndarray] = None

            # --- Thumbnail Erstellung ---
            if generate_thumbnails:
                try:
                    real_crop: np.ndarray = frame[y1:y2, x1:x2]
                    TARGET_H: int = 200
                    crop_h, crop_w = real_crop.shape[:2]
                    if crop_h > 0 and crop_w > 0:
                        scale: float = TARGET_H / crop_h
                        target_w: int = int(crop_w * scale)
                        real_rs: np.ndarray = cv2.resize(real_crop, (target_w, TARGET_H), interpolation=cv2.INTER_LINEAR)
                        cv2.rectangle(real_rs, (0, 0), (target_w - 1, TARGET_H - 1), (0, 255, 0), 2)

                        skel_rs: np.ndarray = np.zeros_like(real_rs)
                        kps_small: np.ndarray = local_kps_array.copy()
                        kps_small[:, 0] = (kps_small[:, 0] - x1) * scale
                        kps_small[:, 1] = (kps_small[:, 1] - y1) * scale

                        self._draw_skeleton_fast(skel_rs, kps_small)
                        self._draw_skeleton_fast(real_rs, kps_small)
                        combined_thumbnail = np.hstack([skel_rs, real_rs])
                except Exception:
                    pass

            detections.append({
                "id": person_counter,
                "bbox": global_bbox,
                "keypoints": rich_keypoints,
                "thumbnail": combined_thumbnail,
                "status": status_display,
                "confidence": conf,
                "color": color
            })
            person_counter += 1

        return detections

    def _draw_skeleton_fast(self, canvas: np.ndarray, kps: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        for i in range(len(self.SKELETON_CONNECTIONS)):
            idx_a: int
            idx_b: int
            idx_a, idx_b = self.SKELETON_CONNECTIONS[i]
            if idx_a >= len(kps) or idx_b >= len(kps): continue
            pt_a: tuple[int, int] = (int(kps[idx_a][0]), int(kps[idx_a][1]))
            pt_b: tuple[int, int] = (int(kps[idx_b][0]), int(kps[idx_b][1]))
            if (0 <= pt_a[0] < w and 0 <= pt_a[1] < h and 0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
                cv2.line(canvas, pt_a, pt_b, (255, 255, 0), 2, cv2.LINE_AA)
        for pt in kps:
            pt_int: tuple[int, int] = (int(pt[0]), int(pt[1]))
            if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
                cv2.circle(canvas, pt_int, 3, (255, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]], scale: float = 1.0) -> np.ndarray:
        if not detections: return frame
        for det in detections:
            bbox: Optional[list[int]] = det.get("bbox")
            if bbox is None or len(bbox) < 4: continue
            try:
                x1: int
                y1: int
                x2: int
                y2: int
                x1, y1 = int(round(float(bbox[0]) * scale)), int(round(float(bbox[1]) * scale))
                x2, y2 = int(round(float(bbox[2]) * scale)), int(round(float(bbox[3]) * scale))
                color: tuple[int, int, int] = det.get("color", (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text: str = f"ID:{det.get('id', '?')} | {str(det.get('status', '')).split(' ')[0]}"
                text_y: int = y1 - 10 if y1 > 20 else y1 + 20
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(text_y - th - 5)), (int(x1 + tw), int(text_y + 5)), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (int(x1), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception:
                pass
        return frame