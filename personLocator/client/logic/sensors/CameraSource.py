import platform

import cv2
import numpy as np
from typing import Tuple, Union, Optional


class CameraSource:
    """
    Kapselt den Zugriff auf die physische Kamera (OpenCV).
    Verarbeitet Hardware-Parameter wie Index und Auflösung.
    """

    def __init__(self, index: int, resolution: Union[str, list, tuple, str]):
        self.index = index
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Öffnet die Kamera und setzt Parameter (Plattformunabhängig)."""

        backend = cv2.CAP_ANY  # Standardfall
        if platform.system() == "Windows":
            backend = cv2.CAP_MSMF
        elif platform.system() == "Linux":
            backend = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.index, backend)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            return False

        self._apply_resolution()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Liest einen Frame."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Gibt Ressourcen frei."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _apply_resolution(self):
        """Parst die Auflösung (String '640x480' oder Liste [640, 480]) und setzt sie."""
        width, height = None, None
        res = self.resolution

        # Fall  "640x480"
        if isinstance(res, str) and "x" in res:
            try:
                width, height = map(int, res.split("x"))
            except ValueError:
                pass

        # Fall Liste/Tuple [640, 480]
        elif isinstance(res, (list, tuple)) and len(res) == 2:
            width, height = res[0], res[1]

        if width and height:
            # 1. Hardware-Kompression erzwingen (MJPEG)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # 2. Auflösung setzen
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # 3. Framerate explizit von der Hardware anfordern
            self.cap.set(cv2.CAP_PROP_FPS, 30)