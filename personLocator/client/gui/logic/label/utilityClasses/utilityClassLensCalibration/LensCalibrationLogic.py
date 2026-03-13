import cv2
import numpy as np
from typing import Tuple, List, Optional


class LensCalibrationLogic:
    """
    Kapselt die gesamte Logik für die intrinsische Kamera-Kalibrierung (Schachbrett).
    """

    def __init__(self, checkerboard_size: Tuple[int, int] = (8, 6), square_size_mm: float = 25.0):
        self.captured_images = []
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size_mm

        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.captured_images_count: int = 0

        self.image_shape: Optional[Tuple[int, int]] = None

    def process_live_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """Findet die Schachbrett-Ecken im Live-Frame und gibt eine visuelle Rückmeldung zurück."""
        if frame is None:
            return frame, False, None

        if self.image_shape is None:
            self.image_shape = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale_down = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=scale_down, fy=scale_down)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

        found, small_corners = cv2.findChessboardCorners(small_gray, self.checkerboard_size, flags)

        display_frame = frame.copy()

        if found:
            corners = (small_corners / scale_down).astype(np.float32)

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners_refined, found)
            return display_frame, True, corners_refined

        return display_frame, False, None

    def capture_calibration_frame(self, corners_refined: np.ndarray, current_image: np.ndarray) -> int:
        """Speichert die Ecken UND das aktuelle Bild für die spätere Galerie."""
        if corners_refined is not None and current_image is not None:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            self.captured_images.append(current_image.copy())
            self.captured_images_count += 1

        return self.captured_images_count

    def calculate_camera_matrix(self) -> Tuple[bool, np.ndarray, np.ndarray, list]:
        """Berechnet die Verzerrung und den Rückprojektionsfehler für jedes Bild."""
        if self.captured_images_count < 10 or self.image_shape is None:
            return False, np.array([]), np.array([]), []

        print(f"Berechne Matrix aus {self.captured_images_count} Bildern...")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_shape[::-1], None, None
        )

        if ret:
            per_image_errors = []
            for i in range(len(self.objpoints)):
                # Wo denkt das Modell, dass die Ecken auf dem Foto sein müssten?
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)

                # Wie weit ist das von den echten, gefundenen Ecken entfernt? (L2-Norm)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                per_image_errors.append(error)

            total_error = sum(per_image_errors) / len(per_image_errors)
            print(f"Kalibrierung erfolgreich! Durchschnittlicher Fehler: {total_error:.2f} Pixel")

            return True, mtx, dist, per_image_errors
        else:
            return False, np.array([]), np.array([]), []