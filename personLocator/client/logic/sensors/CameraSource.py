import platform

import cv2
import numpy as np
from typing import Tuple, Union, Optional, Any


class CameraSource:
    """
    Kapselt den Zugriff auf die physische Kamera (OpenCV).
    Verarbeitet Hardware-Parameter wie Index und Auflösung.
    """

    def __init__(self, index: int, resolution: Union[str, list[int], tuple[int, int]]) -> None:
        self.index: int = index
        self.resolution: Union[str, list[int], tuple[int, int]] = resolution
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Öffnet die Kamera und setzt Parameter (Plattformunabhängig)."""

        backend: int = cv2.CAP_ANY
        if platform.system() == "Windows":
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif platform.system() == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for b in backends:
            self.cap = cv2.VideoCapture(self.index, b)
            if self.cap.isOpened():
                break

        if not self.cap or not self.cap.isOpened():
            return False

        self._apply_resolution()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Liest einen Frame."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Gibt Ressourcen frei."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _apply_resolution(self) -> None:
        """Parst die Auflösung (String '640x480' oder Liste [640, 480]) und setzt sie."""
        if self.cap is None:
            return

        fourcc: int = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        width: Optional[int] = None
        height: Optional[int] = None
        res: Union[str, list[int], tuple[int, int]] = self.resolution

        if isinstance(res, str) and "x" in res:
            try:
                width, height = map(int, res.split("x"))
            except ValueError:
                pass

        elif isinstance(res, (list, tuple)) and len(res) == 2:
            width, height = int(res[0]), int(res[1])

        if width and height:
            # Auflösung muss nach dem Codec gesetzt werden, sonst ignoriert MSMF sie.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fourcc: int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        decoded_fourcc: str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        actual_w: float = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h: float = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps: float = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"🎥 [Hardware Check] Kamera läuft auf: {actual_w}x{actual_h} @ {actual_fps} FPS | Codec: {decoded_fourcc}")