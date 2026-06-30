import numpy as np
from typing import Tuple


class FabrikSolver:
    """
    Hochoptimierte Implementierung des FABRIK-Algorithmus.
    Inklusive 'Smooth Soft-IK' für organisches Gehen OHNE Chicken-Legs im Stand.
    """

    @staticmethod
    def solve_chains_batch(
            p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
            l1: np.ndarray, l2: np.ndarray,
            iterations: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        p0, p1, p2: NumPy Arrays (N, 3) -> Root, Mid, End
        l1, l2: NumPy Arrays (N, 1) -> Längen Knochen 1 und 2
        """
        init_p0: np.ndarray = np.copy(p0)
        target_p2: np.ndarray = np.copy(p2)
        max_reach: np.ndarray = l1 + l2

        def normalize_batch(v: np.ndarray) -> np.ndarray:
            norms: np.ndarray = np.linalg.norm(v, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1e-6, norms)
            return v / norms

        for _ in range(iterations):
            p2 = target_p2
            dir1: np.ndarray = p1 - p2
            p1 = p2 + normalize_batch(dir1) * l2

            dir0: np.ndarray = p0 - p1
            p0 = p1 + normalize_batch(dir0) * l1

            p0 = init_p0
            dir0 = p1 - p0
            p1 = p0 + normalize_batch(dir0) * l1

            dir1 = p2 - p1
            p2 = p1 + normalize_batch(dir1) * l2

        # Smooth Blending zwischen FABRIK-Output und gestreckter Lage,
        # damit die Gliedmaßen nicht abrupt knicken bei Vollausstreckung.
        target_dist: np.ndarray = np.linalg.norm(target_p2 - init_p0, axis=1, keepdims=True)
        ratio: np.ndarray = np.where(max_reach > 1e-6, target_dist / max_reach, 0.0)

        mask_elbow: np.ndarray = (ratio > 0.85) & (target_dist > 1e-6)

        if np.any(mask_elbow):
            safe_dist: np.ndarray = np.where(target_dist < 1e-6, 1e-6, target_dist)
            direction: np.ndarray = (target_p2 - init_p0) / safe_dist

            t_elbow: np.ndarray = (ratio - 0.85) / 0.15
            t_elbow = np.clip(t_elbow, 0.0, 1.0) ** 0.5

            p1_straight: np.ndarray = init_p0 + direction * l1
            p1 = np.where(mask_elbow, p1 * (1.0 - t_elbow) + p1_straight * t_elbow, p1)

            mask_hand: np.ndarray = (ratio > 0.90) & (target_dist > 1e-6)
            if np.any(mask_hand):
                t_hand: np.ndarray = (ratio - 0.90) / 0.10
                t_hand = np.clip(t_hand, 0.0, 1.0)

                p2_straight: np.ndarray = init_p0 + direction * max_reach
                p2 = np.where(mask_hand, p2 * (1.0 - t_hand) + p2_straight * t_hand, p2)

        return p0, p1, p2