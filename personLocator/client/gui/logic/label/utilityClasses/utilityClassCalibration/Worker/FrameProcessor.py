import cv2
import numpy as np


class FrameProcessor:
    """Utility für die Vorverarbeitung von Kamera-Frames."""

    @staticmethod
    def apply_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
        """Dreht den Frame basierend auf dem Winkel."""
        if frame is None or angle == 0: return frame
        try:
            if angle == 90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if angle == 180: return cv2.rotate(frame, cv2.ROTATE_180)
            if angle == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return frame
        except Exception:
            return frame

    @staticmethod
    def apply_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Führt einen digitalen Center-Zoom durch."""
        if zoom_factor <= 1.0 or frame is None: return frame
        h, w = frame.shape[:2]
        nw, nh = int(w / zoom_factor), int(h / zoom_factor)
        sx, sy = (w - nw) // 2, (h - nh) // 2
        cropped = frame[sy:sy + nh, sx:sx + nw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)