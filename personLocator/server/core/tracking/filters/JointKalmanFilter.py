import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Any


class JointKalmanFilter:
    """
    Auf absolute Akkuratheit getrimmter Kalman-Filter mit Reibung.
    HOCHOPTIMIERT: Nutzt Matrix-Caching und Zero-Copy Slicing, um Python-Overhead zu eliminieren.
    """

    def __init__(self) -> None:
        # 6 Zustände (x, y, z, vx, vy, vz), 3 Messwerte (x, y, z)
        self.kf: cv2.KalmanFilter = cv2.KalmanFilter(6, 3)

        # Übergangsmatrix mit Reibung (gedämpfte Geschwindigkeit)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0.5, 0, 0],
            [0, 1, 0, 0, 0.5, 0],
            [0, 0, 1, 0, 0, 0.5],
            [0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0.5]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        self._eye6: np.ndarray = np.eye(6, dtype=np.float32)
        self._eye3: np.ndarray = np.eye(3, dtype=np.float32)

        self.kf.processNoiseCov = self._eye6 * 0.5
        self.kf.measurementNoiseCov = self._eye3 * 0.01
        self.kf.errorCovPost = self._eye6 * 1.0

        self.initialized: bool = False
        self.missed_frames: int = 0

    def predict(self) -> Optional[np.ndarray]:
        """Sagt die nächste Position auf Basis der aktuellen Geschwindigkeit voraus."""
        if not self.initialized:
            return None

        prediction: np.ndarray = self.kf.predict()
        self.missed_frames += 1

        return prediction[:3, 0].copy()

    def update_with_stats(self, measurement_3d: np.ndarray, confidence: float = 1.0, heavy_smoothing: bool = False) -> Tuple[np.ndarray, int, float]:
        """
        Korrigiert die Vorhersage mit dem neuen Kamera-Messwert.
        """
        meas_np: np.ndarray = measurement_3d.reshape(3, 1)

        if not self.initialized:
            self.reset_to(measurement_3d)
            return measurement_3d, 0, 0.0

        pred_pos: np.ndarray = self.kf.statePre[:3, 0]
        jump_distance: float = float(np.linalg.norm(pred_pos - measurement_3d))

        if 10.0 < jump_distance <= 50.0:
            dynamic_process_noise: float = 0.5 + (jump_distance * 0.1)
            self.kf.processNoiseCov = self._eye6 * dynamic_process_noise
        else:
            self.kf.processNoiseCov = self._eye6 * 0.5

        safe_conf: float = max(0.0, min(1.0, float(confidence)))
        dynamic_noise: float = 0.05 + 10.0 * ((1.0 - safe_conf) ** 3)
        
        if heavy_smoothing:
            dynamic_noise *= 15.0
            
        self.kf.measurementNoiseCov = self._eye3 * dynamic_noise

        self.kf.correct(meas_np)
        self.missed_frames = 0

        smoothed_pos: np.ndarray = self.kf.statePost[:3, 0].copy()

        smoothing_cm: float = float(np.linalg.norm(measurement_3d - smoothed_pos))

        return smoothed_pos, 0, smoothing_cm

    def reset_to(self, pos: np.ndarray) -> None:
        """Setzt den Filter hart auf eine neue Position (löscht Geschwindigkeit)."""
        state: np.ndarray = np.zeros((6, 1), dtype=np.float32)
        state[:3, 0] = pos
        self.kf.statePost = state
        self.initialized = True
        self.missed_frames = 0


class SkeletonKalmanTracker:
    """Verwaltet einen unabhängigen dynamischen Kalman-Filter pro Körpergelenk."""

    def __init__(self) -> None:
        self.joints: Dict[int, JointKalmanFilter] = {i: JointKalmanFilter() for i in range(17)}
        self.first_seen: float = 0.0

    def process_frame(self, current_3d_joints: Dict[int, np.ndarray], confidences: Optional[Dict[int, float]] = None, heavy_smoothing: bool = False) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Verarbeitet ein einzelnes Frame und glättet alle vorhandenen Gelenke.
        """
        smoothed_skeleton: Dict[int, np.ndarray] = {}
        stats: Dict[str, Any] = {'blocked': 0, 'smoothing_cm': 0.0}

        if confidences is None:
            confidences = {}

        for j_id, kf in self.joints.items():
            predicted: Optional[np.ndarray] = kf.predict()

            if j_id in current_3d_joints:
                conf: float = confidences.get(j_id, 1.0)
                smoothed_pos: np.ndarray
                blocked: int
                smooth_dist: float
                smoothed_pos, blocked, smooth_dist = kf.update_with_stats(current_3d_joints[j_id], confidence=conf, heavy_smoothing=heavy_smoothing)
                smoothed_skeleton[j_id] = smoothed_pos

                stats['blocked'] += blocked
                stats['smoothing_cm'] += smooth_dist

            elif predicted is not None and kf.missed_frames < 2:
                smoothed_skeleton[j_id] = predicted

        return smoothed_skeleton, stats