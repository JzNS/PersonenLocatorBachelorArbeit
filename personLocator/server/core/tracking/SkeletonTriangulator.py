import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union
from numba import njit
from concurrent.futures import ThreadPoolExecutor

from server.core.math.TriangulationMath import TriangulationMath


@njit(fastmath=True)
def fast_wls_solve(dirs: np.ndarray, cams: np.ndarray, confs: np.ndarray) -> tuple[np.ndarray, bool]:
    N: int = dirs.shape[0]

    S: np.ndarray = np.zeros((3, 3), dtype=np.float32)
    C: np.ndarray = np.zeros(3, dtype=np.float32)
    I: np.ndarray = np.eye(3, dtype=np.float32)

    for i in range(N):
        d: np.ndarray = dirs[i]
        c: np.ndarray = cams[i]
        w: float = confs[i]

        d_dT: np.ndarray = np.empty((3, 3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                d_dT[row, col] = d[row] * d[col]

        W_i: np.ndarray = w * (I - d_dT)
        S += W_i

        for row in range(3):
            val: float = 0.0
            for col in range(3):
                val += W_i[row, col] * c[col]
            C[row] += val

    try:
        pos: np.ndarray = np.linalg.solve(S, C)
        return pos, True
    except Exception:
        return np.zeros(3, dtype=np.float32), False


class SkeletonTriangulator:
    """
    Trianguliert 3D-Skelett-Positionen aus gruppierten Kamera-Clustern.

    Unterstützt zwei Modi:
      "lm"  — Levenberg-Marquardt Optimierung (genauer, langsamer)
      "wls" — Gewichteter Least-Squares Solver (JETZT MIT NUMBA-TURBO)
    """

    Y_MIN: float = -25.0
    Y_MAX: float = 260.0

    def __init__(self, triangulation_mode: str = "wls") -> None:
        self.triangulation_mode: str = triangulation_mode

    def triangulate(
            self,
            clusters: list[dict[str, Any]],
            joint_filter: Optional[dict[int, bool]] = None,
    ) -> list[dict[str, Any]]:
        pre_fusion_data: list[dict[str, Any]] = []

        for cluster_dict in clusters:
            cluster_persons: list[dict[str, Any]] = list(cluster_dict.values())
            result: Optional[dict[str, Any]] = self._triangulate_cluster(cluster_persons, joint_filter)
            if result:
                pre_fusion_data.append(result)

        return pre_fusion_data

    def _triangulate_cluster(
            self,
            cluster_persons: list[dict[str, Any]],
            joint_filter: Optional[dict[int, bool]],
    ) -> Optional[dict[str, Any]]:
        temp_skel_3d: dict[int, np.ndarray] = {}
        temp_skel_conf: dict[int, float] = {}

        target_z: Optional[float] = self._estimate_floor_z(cluster_persons)

        def process_joint(j_id: int) -> tuple[int, Optional[np.ndarray], list[dict[str, Any]]]:
            raw_rays: list[dict[str, Any]] = [
                cp["calculated_rays"][j_id]
                for cp in cluster_persons
                if j_id in cp.get("calculated_rays", {})
            ]

            pos: Optional[np.ndarray] = None
            if len(raw_rays) >= 2:
                pos = self._triangulate_joint_multi(j_id, raw_rays)
            elif len(raw_rays) == 1:
                pos = self._triangulate_joint_single(raw_rays[0], target_z)
            
            return j_id, pos, raw_rays

        joint_ids = [j for j in range(17) if not (joint_filter and not joint_filter.get(j, True))]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_joint, joint_ids))

        for j_id, pos, raw_rays in results:
            if pos is not None and self.Y_MIN <= pos[1] <= self.Y_MAX:
                temp_skel_3d[j_id] = pos
                
                conf_values: list[float] = []
                for r in raw_rays:
                    c = float(r.get("conf", 1.0))
                    if r.get("camera_type") == "Top-Down":
                        c *= 0.8
                    conf_values.append(c)
                
                conf_values = sorted(conf_values)
                temp_skel_conf[j_id] = float(np.mean(conf_values[-2:])) if len(conf_values) >= 2 else float(
                    conf_values[0])

        if not temp_skel_3d:
            return None

        # 2D Reprojektionsfehler berechnen
        total_error = 0.0
        error_count = 0
        for j_id, pos_3d in temp_skel_3d.items():
            for p in cluster_persons:
                if j_id in p.get("calculated_rays", {}):
                    ray = p["calculated_rays"][j_id]
                    if ray.get("P_matrix") is not None:
                        p_hom = np.append(pos_3d, 1.0)
                        pt_proj_hom = np.dot(ray["P_matrix"], p_hom)
                        if abs(pt_proj_hom[2]) > 1e-6:
                            pt_proj = pt_proj_hom[:2] / pt_proj_hom[2]
                            orig_pt = ray["pt_2d"]
                            err = float(np.linalg.norm(pt_proj - orig_pt))
                            total_error += err
                            error_count += 1
        
        reprojection_error = total_error / error_count if error_count > 0 else 0.0

        centroid: np.ndarray = np.mean(list(temp_skel_3d.values()), axis=0)
        colors: dict[str, Any] = {}
        for p in cluster_persons:
            colors.update(p.get("metrics", {}).get("joint_colors", {}))

        return {
            "skel": temp_skel_3d,
            "conf": temp_skel_conf,
            "centroid": centroid,
            "cluster_persons": cluster_persons,
            "id": -1,
            "colors": colors,
            "reprojection_error": reprojection_error
        }

    def _triangulate_joint_multi(
            self,
            j_id: int,
            raw_rays: list[dict[str, Any]],
    ) -> Optional[np.ndarray]:
        dir1: np.ndarray = raw_rays[0]["dir"]
        dir2: np.ndarray = raw_rays[1]["dir"]
        is_highly_uncertain: bool = float(np.abs(np.dot(dir1, dir2))) > 0.96

        if self.triangulation_mode == "lm" and all(r.get("P_matrix") is not None for r in raw_rays):
            return self._solve_lm(raw_rays, is_highly_uncertain)
        else:
            return self._solve_wls(raw_rays, is_highly_uncertain)

    def _solve_lm(
            self,
            raw_rays: list[dict[str, Any]],
            is_highly_uncertain: bool,
    ) -> Optional[np.ndarray]:
        P_mats: list[np.ndarray] = [r["P_matrix"] for r in raw_rays]
        pts_2d: list[np.ndarray] = [r["pt_2d"] for r in raw_rays]
        confs: list[float] = [float(r.get("conf", 1.0)) for r in raw_rays]
        try:
            pos: np.ndarray = TriangulationMath.triangulate_point_optimized(P_mats, pts_2d, confs)
            if is_highly_uncertain and float(np.linalg.norm(pos - raw_rays[0]["cam_pos"])) > 1000.0:
                return raw_rays[0].get("fallback_pos")
            return pos
        except Exception as e:
            print(f"[SkeletonTriangulator] LM Error J{raw_rays[0].get('cam_name', '?')}: {e}")
            return raw_rays[0].get("fallback_pos")

    def _solve_wls(
            self,
            raw_rays: list[dict[str, Any]],
            is_highly_uncertain: bool,
    ) -> Optional[np.ndarray]:
        """
        Nutzt nun den extrem schnellen, präkompilierten Numba-Solver.
        """
        # WICHTIG: Alles explizit in float32 umwandeln, damit Numba nicht abstürzt!
        dirs: np.ndarray = np.array([r["dir"] for r in raw_rays], dtype=np.float32)
        cams: np.ndarray = np.array([r["cam_pos"] for r in raw_rays], dtype=np.float32)
        confs: np.ndarray = np.array([float(r.get("conf", 1.0)) for r in raw_rays], dtype=np.float32)

        if is_highly_uncertain:
            confs = confs * 0.1

        pos: np.ndarray
        success: bool
        pos, success = fast_wls_solve(dirs, cams, confs)

        if not success or (is_highly_uncertain and float(np.linalg.norm(pos - cams[0])) > 1000.0):
            return raw_rays[0].get("fallback_pos")

        return pos

    @staticmethod
    def _triangulate_joint_single(
            ray: dict[str, Any],
            target_z: Optional[float],
    ) -> Optional[np.ndarray]:
        if target_z is not None and abs(ray["dir"][2]) > 1e-6:
            t_z: float = (target_z - ray["cam_pos"][2]) / ray["dir"][2]
            if t_z > 0:
                return ray["cam_pos"] + t_z * ray["dir"]

        fallback: Optional[np.ndarray] = ray.get("fallback_pos")
        return np.array(fallback) if fallback is not None else None

    @staticmethod
    def _estimate_floor_z(cluster_persons: list[dict[str, Any]]) -> Optional[float]:
        for p in cluster_persons:
            valid_kps: list[dict[str, Any]] = [k for k in p.get("keypoints", []) if float(k.get("c", 0)) > 0.4]
            if not valid_kps:
                continue
            lowest_kp: dict[str, Any] = max(valid_kps, key=lambda k: float(k.get("y", 0)))
            ray: Optional[dict[str, Any]] = p.get("calculated_rays", {}).get(lowest_kp["id"])
            if ray and abs(ray["dir"][1]) > 1e-6:
                t_floor: float = -ray["cam_pos"][1] / ray["dir"][1]
                if t_floor > 0:
                    return float((ray["cam_pos"] + t_floor * ray["dir"])[2])
        return None