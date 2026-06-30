import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any, List


class AcausalCurveFilter:
    """
    Ein akausaler Sliding-Window-Filter (Savitzky-Golay), der 2 Frames in die Zukunft blickt.
    Nutzt perfekte polynomische Glättung für absolut ruckelfreie, natürliche Bewegungen.

    Latenz: Exakt 2 Frames (ca. 66ms bei 30 FPS).
    Rechenleistung: Minimal (nur einfache Vektor-Multiplikationen).
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("Window size muss ungerade und mindestens 3 sein (z.B. 5, 7, 9).")

        self.window_size: int = window_size
        self.half_window: int = window_size // 2

        self.history: Dict[int, deque[np.ndarray]] = {}

        if window_size == 5:
            self.weights: np.ndarray = np.array([-3, 12, 17, 12, -3], dtype=np.float32) / 35.0
        elif window_size == 7:
            self.weights = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=np.float32) / 21.0
        else:
            self.weights = np.ones(window_size, dtype=np.float32) / float(window_size)

    def process_frame(self, skeleton_3d: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
        """
        Nimmt das aktuelle Frame auf und gibt das geglättete Frame aus der Vergangenheit (N-2) zurück.
        """
        smoothed_skeleton: Dict[int, np.ndarray] = {}
        stats: Dict[str, float] = {'smoothing_cm': 0.0, 'latency_frames': float(self.half_window)}

        if not skeleton_3d:
            return {}, stats

        for j_id, pos in skeleton_3d.items():
            if j_id not in self.history:
                self.history[j_id] = deque([pos.copy() for _ in range(self.window_size)], maxlen=self.window_size)
            else:
                self.history[j_id].append(pos.copy())

        for j_id, history_buffer in self.history.items():
            if len(history_buffer) == self.window_size:
                pts_array: np.ndarray = np.array(history_buffer, dtype=np.float32)

                smoothed_pos: np.ndarray = np.dot(self.weights, pts_array)
                smoothed_skeleton[j_id] = smoothed_pos

                raw_mid_pos: np.ndarray = pts_array[self.half_window]
                stats['smoothing_cm'] += float(np.linalg.norm(smoothed_pos - raw_mid_pos))
            else:
                smoothed_skeleton[j_id] = history_buffer[len(history_buffer) // 2]

        return smoothed_skeleton, stats

    def reset(self) -> None:
        """Leert den Ringpuffer (z.B. bei einem Szenenwechsel oder Teleportation)."""
        self.history.clear()