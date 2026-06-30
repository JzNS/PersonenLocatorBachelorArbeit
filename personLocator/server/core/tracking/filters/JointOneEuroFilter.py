import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, Union, List


class SkeletonOneEuroTracker:
    """
    Hochoptimierter Vektor-1-Euro-Filter.
    Ersetzt 17 einzelne Python-Klassen durch EINE schnelle C-Matrix-Operation!
    """

    def __init__(self, mincutoff: float = 0.005, default_beta: float = 0.07, dcutoff: float = 1.0) -> None:
        self.mincutoff: float = mincutoff
        self.default_beta: float = default_beta
        self.dcutoff: float = dcutoff

        self.x_prev: np.ndarray = np.zeros((17, 3), dtype=np.float32)
        self.dx_prev: np.ndarray = np.zeros((17, 3), dtype=np.float32)
        self.t_prev: np.ndarray = np.zeros(17, dtype=np.float64)

        self.initialized: np.ndarray = np.zeros(17, dtype=bool)

        self.first_seen: float = time.time()

    def _smoothing_factor(self, t_e: np.ndarray, cutoff: Union[np.ndarray, float]) -> np.ndarray:
        r: np.ndarray = 2 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def process_frame(self, skeleton_3d: dict[int, np.ndarray], confidences: Optional[dict[int, float]] = None, heavy_smoothing: bool = False) -> tuple[
        dict[int, np.ndarray], dict[str, Any]]:
        now: float = time.time()
        if not skeleton_3d:
            return {}, {"blocked": 0, "smoothing_cm": 0.0}

        if confidences is None:
            confidences = {}

        ids: list[int] = list(skeleton_3d.keys())
        x: np.ndarray = np.array(list(skeleton_3d.values()), dtype=np.float32)  # Shape: (N, 3)

        t_e: np.ndarray = np.full((len(ids), 1), now, dtype=np.float64)
        mask_init: np.ndarray = self.initialized[ids]

        t_e[mask_init, 0] = now - self.t_prev[np.array(ids)[mask_init]]

        base_beta = self.default_beta
        if heavy_smoothing:
            base_beta *= 0.1  # Drastisch reduzieren bei Top-Down (weniger Zappeln)

        betas: np.ndarray = np.full((len(ids), 1), base_beta, dtype=np.float32)
        for i, j_id in enumerate(ids):
            c = confidences.get(j_id, 1.0)
            if c < 0.5:
                betas[i, 0] = base_beta * 0.05

        # Min-Cutoff bei Top-Down ebenfalls drosseln
        actual_mincutoff = self.mincutoff
        if heavy_smoothing:
            actual_mincutoff *= 0.2

        x_hat: np.ndarray = np.empty_like(x)
        new_idx: np.ndarray = np.where(~mask_init)[0]
        if len(new_idx) > 0:
            g_ids: np.ndarray = np.array(ids)[new_idx]
            x_new: np.ndarray = x[new_idx]
            x_hat[new_idx] = x_new

            self.x_prev[g_ids] = x_new
            self.dx_prev[g_ids] = 0.0
            self.t_prev[g_ids] = now
            self.initialized[g_ids] = True

        upd_idx: np.ndarray = np.where(mask_init)[0]
        if len(upd_idx) > 0:
            g_ids: np.ndarray = np.array(ids)[upd_idx]
            x_upd: np.ndarray = x[upd_idx]
            t_upd: np.ndarray = t_e[upd_idx]

            valid: np.ndarray = t_upd > 0.0
            t_safe: np.ndarray = np.where(valid, t_upd, 1.0)

            xp: np.ndarray = self.x_prev[g_ids]
            dxp: np.ndarray = self.dx_prev[g_ids]

            dx: np.ndarray = (x_upd - xp) / t_safe
            alpha_d: np.ndarray = self._smoothing_factor(t_safe, self.dcutoff)
            dx_hat: np.ndarray = alpha_d * dx + (1.0 - alpha_d) * dxp

            dx_norm: np.ndarray = np.linalg.norm(dx_hat, axis=1, keepdims=True)
            cutoff: np.ndarray = actual_mincutoff + betas[upd_idx] * dx_norm

            alpha: np.ndarray = self._smoothing_factor(t_safe, cutoff)
            x_h: np.ndarray = alpha * x_upd + (1.0 - alpha) * xp

            x_hat_final: np.ndarray = np.where(valid, x_h, x_upd)
            x_hat[upd_idx] = x_hat_final

            self.x_prev[g_ids] = x_hat_final
            self.dx_prev[g_ids] = np.where(valid, dx_hat, dxp)
            self.t_prev[g_ids] = now

        smooth_dist: float = 0.0
        if len(upd_idx) > 0:
            diffs = x_upd - x_hat_final
            dists = np.linalg.norm(diffs, axis=1)
            smooth_dist = float(np.sum(dists))

        filtered_skel: dict[int, np.ndarray] = {j_id: x_hat[i] for i, j_id in enumerate(ids)}

        return filtered_skel, {"blocked": 0, "smoothing_cm": smooth_dist}
