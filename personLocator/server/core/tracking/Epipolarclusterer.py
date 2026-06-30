import numpy as np
from typing import Any, List, Dict, Tuple, Optional

from server.rendering.DrawingUtils import DrawingUtils


class EpipolarClusterer:
    """
    Gruppiert Personen aus verschiedenen Kameras zu 3D-Clustern.

    Entscheidungskriterien:
      1. Epipolare Distanz der gemeinsamen Sichtstrahlen
      2. Farb-Ähnlichkeit der übereinstimmenden Gelenke
    """

    def __init__(
        self,
        max_epipolar_error_cm: float  = 35.0,
        epipolar_ignore_threshold: float = 1000.0,
        color_match_threshold: float  = 0.4,
        min_joint_match_ratio: float  = 0.30,
    ) -> None:
        self.max_epipolar_error_cm: float   = max_epipolar_error_cm
        self.epipolar_ignore_threshold: float = epipolar_ignore_threshold
        self.color_match_threshold: float   = color_match_threshold
        self.min_joint_match_ratio: float   = min_joint_match_ratio

        self._epipolar_error_sum: float   = 0.0
        self._epipolar_error_count: int = 0
        self._ghost_count: int          = 0

    def compute(self, persons: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
        """
        Nimmt eine Liste von Personen (mit 'calculated_rays') und gibt eine
        Liste von Kamera-Clustern zurück.

        Ein Cluster ist ein Dict { cam_name: person_dict }.
        """
        clusters: list[dict[str, dict[str, Any]]] = []

        for p in persons:
            cam: Optional[str] = p.get("cam_name")
            if not cam or not cam.startswith("CAMERA_"):
                continue

            best_cluster, best_score = self._find_best_cluster(p, cam, clusters)

            if best_cluster is not None:
                best_cluster[cam] = p
            else:
                clusters.append({cam: p})

        return clusters

    def get_stats(self) -> dict[str, Any]:
        avg_err: float = (
            self._epipolar_error_sum / self._epipolar_error_count
            if self._epipolar_error_count > 0 else 0.0
        )
        return {
            "epipolar_ghosts":      self._ghost_count,
            "epipolar_error_avg":   avg_err,
            "epipolar_error_sum":   self._epipolar_error_sum,
            "epipolar_error_count": self._epipolar_error_count,
        }

    def reset_stats(self) -> None:
        self._epipolar_error_sum   = 0.0
        self._epipolar_error_count = 0
        self._ghost_count          = 0

    def _find_best_cluster(
        self,
        p: dict[str, Any],
        cam: str,
        clusters: list[dict[str, dict[str, Any]]],
    ) -> tuple[Optional[dict[str, dict[str, Any]]], float]:
        """Gibt das beste passende Cluster (und seinen Score) zurück."""
        best_cluster: Optional[dict[str, dict[str, Any]]] = None
        best_score: float   = -1.0
        p_rays: dict[int, Any] = p.get("calculated_rays", {})

        for cluster_dict in clusters:
            if cam in cluster_dict:
                continue

            is_top_down = (p.get("camera_type") == "Top-Down")
            
            score: float = self._score_cluster_match(p_rays, cluster_dict)
            
            if is_top_down:
                score *= 1.2
            
            if score >= self.min_joint_match_ratio and score > best_score:
                best_score   = score
                best_cluster = cluster_dict

        return best_cluster, best_score

    def _score_cluster_match(
        self,
        p_rays: dict[int, Any],
        cluster_dict: dict[str, dict[str, Any]],
    ) -> float:
        """
        Berechnet den Übereinstimmungs-Score zwischen einer Person und einem Cluster.
        Score = matching_joints / valid_joints  (0..1)
        """
        valid_joints: int   = 0
        matching_joints: int = 0

        for ref_p in cluster_dict.values():
            ref_rays: dict[int, Any]   = ref_p.get("calculated_rays", {})
            common_ids: list[int] = [j for j in range(17) if j in p_rays and j in ref_rays]
            if not common_ids:
                continue

            dists: np.ndarray = self._batch_epipolar_distances(p_rays, ref_rays, common_ids)

            for idx, j_id in enumerate(common_ids):
                ep_dist: float = float(dists[idx])
                valid_joints += 1

                if ep_dist < self.epipolar_ignore_threshold:
                    self._epipolar_error_sum   += ep_dist
                    self._epipolar_error_count += 1

                if ep_dist < self.max_epipolar_error_cm:
                    c1: Any = p_rays[j_id]["color"]
                    c2: Any = ref_rays[j_id]["color"]
                    col_dist: float = 0.0
                    if c1 is not None and c2 is not None:
                        col_dist = DrawingUtils.calculate_color_distance({j_id: c1}, {j_id: c2})
                    if col_dist < self.color_match_threshold:
                        matching_joints += 1
                else:
                    self._ghost_count += 1

        if valid_joints == 0:
            return 0.0
        return matching_joints / valid_joints

    @staticmethod
    def _batch_epipolar_distances(
        p_rays: dict[int, Any],
        ref_rays: dict[int, Any],
        common_ids: list[int],
    ) -> np.ndarray:
        """Vektorisierte Berechnung der Epipolar-Distanzen für alle gemeinsamen Gelenke."""
        o1: np.ndarray = np.array([p_rays[j]["cam_pos"] for j in common_ids],   dtype=np.float32)
        d1: np.ndarray = np.array([p_rays[j]["dir"]     for j in common_ids],   dtype=np.float32)
        o2: np.ndarray = np.array([ref_rays[j]["cam_pos"] for j in common_ids], dtype=np.float32)
        d2: np.ndarray = np.array([ref_rays[j]["dir"]     for j in common_ids], dtype=np.float32)
        return DrawingUtils.calculate_epipolar_distance_batch(o1, d1, o2, d2)