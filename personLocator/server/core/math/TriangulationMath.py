import cv2
import numpy as np
from server.core.logger import get_logger
from scipy.optimize import least_squares
from typing import List, Dict, Any, Union, Tuple
from server.core.exceptions import TriangulationError

logger = get_logger("server.math.triangulation")



class TriangulationMath:
    """
    Berechnet echte 3D-Koordinaten aus mehreren Kameraperspektiven
    durch das Schneiden von Sichtstrahlen (Triangulation).
    """

    @staticmethod
    def reprojection_error(X_3d: np.ndarray, P_matrices: np.ndarray, points_2d: np.ndarray,
                           confidences: np.ndarray) -> np.ndarray:
        """Vektorisierte Kostenfunktion in reinem C++ (über NumPy). Umgeht den Python-GIL!"""
        X_homog: np.ndarray = np.append(X_3d, 1.0)

        proj: np.ndarray = P_matrices @ X_homog

        z: np.ndarray = proj[:, 2]
        z[z == 0] = 1e-6

        proj_2d: np.ndarray = proj[:, :2] / z[:, np.newaxis]

        residual: np.ndarray = (proj_2d - points_2d) * confidences[:, np.newaxis]
        return residual.flatten()

    @staticmethod
    def triangulate_point_optimized(P_matrices: list[np.ndarray], points_2d: list[Union[list[float], tuple[float, float], np.ndarray]], confidences: list[float]) -> np.ndarray:
        P_arr: np.ndarray = np.array(P_matrices)
        pts_arr: np.ndarray = np.array(points_2d)
        conf_arr: np.ndarray = np.array(confidences)

        pt1_arr: np.ndarray = np.array([[pts_arr[0][0]], [pts_arr[0][1]]], dtype=np.float64)
        pt2_arr: np.ndarray = np.array([[pts_arr[1][0]], [pts_arr[1][1]]], dtype=np.float64)
        pt4d: np.ndarray = cv2.triangulatePoints(P_arr[0], P_arr[1], pt1_arr, pt2_arr)
        x0: np.ndarray = (pt4d[:3] / pt4d[3]).flatten()

        res = least_squares(
            TriangulationMath.reprojection_error,
            x0,
            args=(P_arr, pts_arr, conf_arr),
            method='lm',
            max_nfev=25,
            ftol=1e-3,
            xtol=1e-3
        )
        return res.x

    @staticmethod
    def get_projection_matrix(camera_matrix: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Erstellt die 3x4 Projektionsmatrix (P) für eine Kamera.
        P = K * [R | t]
        """
        R: np.ndarray
        R, _ = cv2.Rodrigues(rvec)
        Rt: np.ndarray = np.hstack((R, tvec))
        P: np.ndarray = np.dot(camera_matrix, Rt)

        return np.ascontiguousarray(P, dtype=np.float64)

    @staticmethod
    def triangulate_point(P1: np.ndarray, P2: np.ndarray, pt1: Union[tuple[float, float], list[float], np.ndarray], pt2: Union[tuple[float, float], list[float], np.ndarray]) -> np.ndarray:
        """
        Berechnet den 3D-Schnittpunkt von zwei 2D-Punkten aus verschiedenen Kameras.
        """
        pt1_arr: np.ndarray = np.ascontiguousarray([[float(pt1[0])], [float(pt1[1])]], dtype=np.float64)
        pt2_arr: np.ndarray = np.ascontiguousarray([[float(pt2[0])], [float(pt2[1])]], dtype=np.float64)

        pt4d: np.ndarray = cv2.triangulatePoints(P1, P2, pt1_arr, pt2_arr)
        pt3d: np.ndarray = pt4d[:3] / pt4d[3]

        return pt3d.flatten()

    @staticmethod
    def triangulate_skeleton(P1: np.ndarray, P2: np.ndarray, kps1: list[dict[str, Any]], kps2: list[dict[str, Any]], min_conf: float = 0.5) -> dict[int, np.ndarray]:
        """
        Trianguliert alle Gelenke in einem einzigen, massiven OpenCV C++ Aufruf (Vektorisiert!).
        """
        try:
            map1: dict[int, tuple[float, float, float]] = {int(k['id']): (float(k['x']), float(k['y']), float(k['c'])) for k in kps1}
            map2: dict[int, tuple[float, float, float]] = {int(k['id']): (float(k['x']), float(k['y']), float(k['c'])) for k in kps2}

            common_ids: list[int] = [
                j_id for j_id in map1.keys()
                if j_id in map2 and map1[j_id][2] >= min_conf and map2[j_id][2] >= min_conf
            ]

            if not common_ids:
                return {}

            pts1_arr: np.ndarray = np.array([[map1[j][0] for j in common_ids],
                                 [map1[j][1] for j in common_ids]], dtype=np.float64)

            pts2_arr: np.ndarray = np.array([[map2[j][0] for j in common_ids],
                                 [map2[j][1] for j in common_ids]], dtype=np.float64)

            pt4d: np.ndarray = cv2.triangulatePoints(P1, P2, pts1_arr, pts2_arr)

            pt3d: np.ndarray = pt4d[:3, :] / pt4d[3, :]

            return {j_id: pt3d[:, i] for i, j_id in enumerate(common_ids)}
        except Exception as e:
            err = TriangulationError("Skeleton", f"Batch-Fehler: {str(e)}")
            logger.error(str(err))
            return {}