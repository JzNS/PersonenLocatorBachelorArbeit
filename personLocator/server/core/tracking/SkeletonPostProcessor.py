import time
import concurrent.futures
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Callable

from server.core.tracking.filters.JointKalmanFilter import SkeletonKalmanTracker
from server.core.tracking.filters.JointOneEuroFilter import SkeletonOneEuroTracker
from server.core.tracking.filters.SkeletonFABRIK import FabrikSolver
from server.core.tracking.filters.AcausalCurveFilter import AcausalCurveFilter
from server.config.TrackingConfig import TrackingConfig


class SkeletonPostProcessor:
    """
    Pipeline nach der Triangulation:

      1. Temporaler Outlier-Filter   (Sprung-Unterdrückung)
      2. Bewegungsglättung           (1-Euro oder Kalman)
      3. Knochenlängen lernen        (adaptiv, exponentiell)
      4. Inverse Kinematik           (FABRIK oder Classic)
      5. Schulter-/Hüft-Symmetrie   (optional)
      6. Center-of-Mass Limit        (verhindert "Geister-Glieder")
      7. Akausaler Savitzky-Golay    (2-Frame Blick voraus)
      8. Orientierungsschätzung      (Voting: Gesicht + Knie + Bewegung)
      9. Sichtbarkeits-Tag           (is_visible / is_confirmed)
    """

    VISIBILITY_DELAY: float  = 1.5  # Sekunden bis is_visible = True
    CONFIRM_DELAY: float     = 2.5  # Sekunden bis is_confirmed = True

    def __init__(
        self,
        config: TrackingConfig,
        smoothing_mode: str = "one_euro",
        ik_mode: str = "fabrik",
    ) -> None:
        self.config: TrackingConfig        = config
        self.smoothing_mode: str = smoothing_mode
        self.ik_mode: str       = ik_mode

        self._kalman_trackers:  dict[int, SkeletonKalmanTracker]  = {}
        self._one_euro_trackers: dict[int, SkeletonOneEuroTracker] = {}
        self._acausal_trackers: dict[int, AcausalCurveFilter]     = {}
        self._temporal_memory:  dict[int, dict[int, dict[str, Any]]]        = {}
        self._learned_bones:    dict[int, dict[str, float]]       = {}

        self._stats_ik_resizes: int          = 0
        self._stats_ik_correction_cm: float    = 0.0
        self._stats_kalman_blocked: int      = 0
        self._stats_kalman_smoothing_cm: float = 0.0

    def process(self, data: dict[str, Any], now: float) -> None:
        """
        Verändert `data` in-place:
          • data["skel"]         → geglättetes, IK-korrigiertes Skelett
          • data["is_confirmed"] → True wenn Person ≥ 2,5 s sichtbar
          • data["forward_vec"]  → Blickrichtungsvektor
        """
        top_down_influence = any(cp.get("camera_type") == "Top-Down" for cp in data.get("cluster_persons", []))

        if self.ik_mode == "fabrik":
            self._fabrik_pipeline(data, now, heavy_smoothing=top_down_influence)
        else:
            self._classic_pipeline(data, now, heavy_smoothing=top_down_influence)

        person_id: int = data["id"]
        if person_id not in self._temporal_memory:
            self._temporal_memory[person_id] = {}

        if data.get("skel"):
            data["forward_vec"] = self._calculate_orientation(
                data["skel"], self._temporal_memory[person_id]
            )

    def cleanup_person(self, person_id: int) -> None:
        """Entfernt alle Zustandsobjekte einer abgemeldeten Person."""
        for store in (
            self._kalman_trackers, self._one_euro_trackers,
            self._acausal_trackers, self._temporal_memory, self._learned_bones,
        ):
            store.pop(person_id, None)

    def get_learned_bones(self, person_id: int) -> dict[str, float]:
        return self._learned_bones.get(person_id, {})

    def get_stats(self) -> dict[str, Any]:
        avg_smooth = self._stats_kalman_smoothing_cm / max(1, self._stats_ik_resizes + self._stats_kalman_blocked + 1) # Grobe Approximation

        health_score = 100.0 - (avg_smooth * 0.5) - (self._stats_kalman_blocked * 0.1) - (self._stats_ik_resizes * 0.05)
        health_score = max(0.0, min(100.0, health_score))

        return {
            "ik_resizes":           self._stats_ik_resizes,
            "ik_correction_cm":     self._stats_ik_correction_cm,
            "kalman_glitches_blocked": self._stats_kalman_blocked,
            "kalman_smoothing_cm":  self._stats_kalman_smoothing_cm,
            "health_index":         health_score
        }

    def _fabrik_pipeline(self, data: dict[str, Any], now: float, heavy_smoothing: bool = False) -> None:
        skel: dict[int, np.ndarray]       = data["skel"]
        conf: dict[int, float]       = data.get("conf", {})
        person_id: int  = data["id"]

        if person_id not in self._learned_bones:
            self._learned_bones[person_id] = {}
        if person_id not in self._temporal_memory:
            self._temporal_memory[person_id] = {}

        skel = self._apply_outlier_filter(skel, conf, person_id, now)
        skel, tracker = self._apply_smoothing(skel, conf, person_id, heavy_smoothing=heavy_smoothing)
        self._learn_bone_lengths(skel, person_id)
        skel = self._run_fabrik(skel, person_id)

        if data.get("force_symmetry", True):
            self._enforce_symmetry(skel)

        self._apply_com_limit(skel)
        skel = self._apply_acausal(skel, person_id)

        age: float          = time.time() - getattr(tracker, "first_seen", time.time())
        is_visible: bool   = age >= self.VISIBILITY_DELAY
        is_confirmed: bool = age >= self.CONFIRM_DELAY

        if not self._is_skeleton_complete(skel) or not is_visible:
            skel.clear()

        data["skel"]         = skel
        data["is_confirmed"] = is_confirmed

    def _classic_pipeline(self, data: dict[str, Any], now: float, heavy_smoothing: bool = False) -> None:
        skel: dict[int, np.ndarray]      = data["skel"]
        conf: dict[int, float]      = data.get("conf", {})
        person_id: int = data["id"]

        if person_id not in self._learned_bones:
            self._learned_bones[person_id] = {}
        if person_id not in self._temporal_memory:
            self._temporal_memory[person_id] = {}

        self._learn_bone_lengths(skel, person_id)
        self._apply_classic_ik_constraints(skel, person_id)

        if data.get("force_symmetry", True):
            self._enforce_symmetry(skel)

        self._apply_com_limit(skel)
        skel = self._apply_outlier_filter(skel, conf, person_id, now)
        skel, tracker = self._apply_smoothing(skel, conf, person_id, heavy_smoothing=heavy_smoothing)
        skel = self._apply_acausal(skel, person_id)

        age: float          = time.time() - getattr(tracker, "first_seen", time.time())
        is_visible: bool   = age >= self.VISIBILITY_DELAY
        is_confirmed: bool = age >= self.CONFIRM_DELAY

        if not self._is_skeleton_complete(skel) or not is_visible:
            skel.clear()

        data["skel"]         = skel
        data["is_confirmed"] = is_confirmed

    def _apply_outlier_filter(
            self,
            skel: dict[int, np.ndarray],
            conf: dict[int, float],
            person_id: int,
            now: float,
    ) -> dict[int, np.ndarray]:
        """Unterdrückt Ausreißer IN-PLACE (Zero-Allocation)."""
        mem: dict[int, dict[str, Any]] = self._temporal_memory[person_id]

        for j_id in list(skel.keys()):
            current_pos: np.ndarray = skel[j_id]

            if j_id not in mem:
                mem[j_id] = {"pos": current_pos, "outlier_start": 0.0}
                continue

            last_pos: np.ndarray = mem[j_id]["pos"]
            delta: float = float(np.linalg.norm(current_pos - last_pos))

            if delta < 0.4:
                skel[j_id] = last_pos
                continue

            allowed: float = float(self.config.max_frame_delta.get(j_id, 20.0))
            if conf.get(j_id, 1.0) > 0.85:
                allowed *= 3.0

            if delta > allowed:
                if mem[j_id]["outlier_start"] == 0:
                    mem[j_id]["outlier_start"] = now

                if (now - mem[j_id]["outlier_start"]) > self.config.outlier_recovery_time:
                    mem[j_id]["pos"] = current_pos
                    mem[j_id]["outlier_start"] = 0.0
                else:
                    skel[j_id] = last_pos
            else:
                mem[j_id]["pos"] = current_pos
                mem[j_id]["outlier_start"] = 0.0

        return skel

    def _apply_smoothing(
        self,
        skel: dict[int, np.ndarray],
        conf: dict[int, float],
        person_id: int,
        heavy_smoothing: bool = False,
    ) -> tuple[dict[int, np.ndarray], Any]:
        """Wendet 1-Euro-, Kalman-Filter oder keinen Filter an. Gibt (Skelett, tracker) zurück."""
        tracker: Any
        if self.smoothing_mode == "none":
            tracker = self._get_or_create(self._one_euro_trackers, person_id, SkeletonOneEuroTracker)
            return skel, tracker
        elif self.smoothing_mode == "one_euro":
            tracker = self._get_or_create(self._one_euro_trackers, person_id, SkeletonOneEuroTracker)
            skel, s_stats = tracker.process_frame(skel, confidences=conf, heavy_smoothing=heavy_smoothing)
            
            self._stats_kalman_blocked      += s_stats.get("blocked", 0)
            self._stats_kalman_smoothing_cm += s_stats.get("smoothing_cm", 0.0)
        else:
            if person_id not in self._kalman_trackers:
                k_tracker = SkeletonKalmanTracker()
                k_tracker.first_seen = time.time()
                self._kalman_trackers[person_id] = k_tracker
            tracker = self._kalman_trackers[person_id]

            k_stats: dict[str, Any]
            try:
                skel, k_stats = tracker.process_frame(skel, confidences=conf, heavy_smoothing=heavy_smoothing)
            except TypeError:
                skel, k_stats = tracker.process_frame(skel)

            self._stats_kalman_blocked      += k_stats.get("blocked", 0)
            self._stats_kalman_smoothing_cm += k_stats.get("smoothing_cm", 0.0)

        return skel, tracker

    def _learn_bone_lengths(self, skel: dict[int, np.ndarray], person_id: int) -> None:
        """Lernt Knochenlängen adaptiv per exponentiellen gleitenden Mittelwert."""
        bones: dict[str, float] = self._learned_bones[person_id]
        for parent, child, abs_min, abs_max, bone_name in self.config.ik_chain_strict:
            if parent not in skel or child not in skel:
                continue
            dist: float = float(np.linalg.norm(skel[child] - skel[parent]))
            if abs_min <= dist <= abs_max:
                if bone_name not in bones:
                    bones[bone_name] = dist
                else:
                    bones[bone_name] = 0.02 * dist + 0.98 * bones[bone_name]

    def _run_fabrik(
            self,
            skel: dict[int, np.ndarray],
            person_id: int,
    ) -> dict[int, np.ndarray]:
        """Vektorisierter FABRIK-Löser ohne langsamen Thread-Overhead."""
        bones: dict[str, float] = self._learned_bones[person_id]

        limb_chains: list[tuple[int, int, int, str, str, float, float]] = [
            (5, 7, 9, "arm_l_up", "arm_l_down", 30.0, 26.0),
            (6, 8, 10, "arm_r_up", "arm_r_down", 30.0, 26.0),
            (11, 13, 15, "leg_l_up", "leg_l_down", 45.0, 42.0),
            (12, 14, 16, "leg_r_up", "leg_r_down", 45.0, 42.0),
        ]

        valid_limbs: list[tuple[int, int, int]] = []
        p0_list: list[np.ndarray] = []
        p1_list: list[np.ndarray] = []
        p2_list: list[np.ndarray] = []
        l1_list: list[float] = []
        l2_list: list[float] = []

        for chain in limb_chains:
            root_id, mid_id, end_id, b_up, b_down, def_up, def_down = chain
            if all(k in skel for k in [root_id, mid_id, end_id]):
                valid_limbs.append((root_id, mid_id, end_id))

                p0_list.append(skel[root_id])
                p1_list.append(skel[mid_id])
                p2_list.append(skel[end_id])

                l1_list.append(bones.get(b_up, def_up))
                l2_list.append(bones.get(b_down, def_down))

        if not valid_limbs:
            return skel

        p0_arr: np.ndarray = np.array(p0_list, dtype=np.float32)
        p1_arr: np.ndarray = np.array(p1_list, dtype=np.float32)
        p2_arr: np.ndarray = np.array(p2_list, dtype=np.float32)
        l1_arr: np.ndarray = np.array(l1_list, dtype=np.float32).reshape(-1, 1)
        l2_arr: np.ndarray = np.array(l2_list, dtype=np.float32).reshape(-1, 1)

        new_p0: np.ndarray
        new_p1: np.ndarray
        new_p2: np.ndarray
        new_p0, new_p1, new_p2 = FabrikSolver.solve_chains_batch(
            p0_arr, p1_arr, p2_arr, l1_arr, l2_arr, iterations=4
        )

        for i, (root_id, mid_id, end_id) in enumerate(valid_limbs):
            d1 = float(np.linalg.norm(skel[root_id] - new_p0[i]))
            d2 = float(np.linalg.norm(skel[mid_id] - new_p1[i]))
            d3 = float(np.linalg.norm(skel[end_id] - new_p2[i]))
            
            diff_sum = d1 + d2 + d3
            if diff_sum > 0.001:
                self._stats_ik_correction_cm += diff_sum
                self._stats_ik_resizes += 1

            skel[root_id] = new_p0[i]
            skel[mid_id] = new_p1[i]
            skel[end_id] = new_p2[i]

        return skel

    def _apply_classic_ik_constraints(
        self,
        skel: dict[int, np.ndarray],
        person_id: int,
    ) -> None:
        """Classic IK: begrenzt Knochenlängen auf das gelernte Intervall."""
        bones: dict[str, float] = self._learned_bones[person_id]
        for parent, child, abs_min, abs_max, bone_name in self.config.ik_chain_strict:
            if parent not in skel or child not in skel:
                continue
            p_pos: np.ndarray = skel[parent]
            c_pos: np.ndarray = skel[child]
            vec: np.ndarray   = c_pos - p_pos
            dist: float  = float(np.linalg.norm(vec))

            learned: float = bones.get(bone_name, (abs_min + abs_max) / 2.0)
            dyn_min: float = max(abs_min, learned * 0.85)
            dyn_max: float = min(abs_max, learned * 1.15)

            if dist > dyn_max:
                skel[child] = p_pos + (vec / dist) * dyn_max
                self._stats_ik_resizes       += 1
                self._stats_ik_correction_cm += float(dist - dyn_max)
            elif 0.001 < dist < dyn_min:
                skel[child] = p_pos + (vec / dist) * dyn_min
                self._stats_ik_resizes       += 1
                self._stats_ik_correction_cm += float(dyn_min - dist)

    def _enforce_symmetry(self, skel: dict[int, np.ndarray]) -> None:
        """Korrigiert Schulter- und Hüft-Breite auf symmetrische Positionierung."""
        if not all(k in skel for k in [5, 6, 11, 12]):
            return

        def _symmetrize(left_id: int, right_id: int) -> None:
            mid: np.ndarray   = (skel[left_id] + skel[right_id]) / 2.0
            width: float = float(np.linalg.norm(skel[left_id] - skel[right_id]))
            dir_: np.ndarray  = (skel[left_id] - skel[right_id]) / (width + 1e-6)
            skel[left_id]  = mid + dir_ * (width / 2.0)
            skel[right_id] = mid - dir_ * (width / 2.0)

        _symmetrize(5,  6)
        _symmetrize(11, 12)

    def _apply_com_limit(self, skel: dict[int, np.ndarray]) -> None:
        """Beschränkt Gelenke auf den maximalen Abstand vom Massezentrum."""
        anchor_pts: list[np.ndarray] = [skel[k] for k in [0] if k in skel]
        if 5  in skel and 6  in skel: anchor_pts.append((skel[5]  + skel[6])  / 2.0)
        if 11 in skel and 12 in skel: anchor_pts.append((skel[11] + skel[12]) / 2.0)

        if not anchor_pts:
            return

        com: np.ndarray = np.mean(anchor_pts, axis=0)
        for j_id, max_dist in self.config.max_com_reach.items():
            if j_id not in skel:
                continue
            vec: np.ndarray  = skel[j_id] - com
            dist: float = float(np.linalg.norm(vec))
            if dist > max_dist:
                skel[j_id] = com + (vec / dist) * max_dist

    def _apply_acausal(
        self,
        skel: dict[int, np.ndarray],
        person_id: int,
    ) -> dict[int, np.ndarray]:
        """Savitzky-Golay Sliding-Window Filter mit 2-Frame Lookahead."""
        if person_id not in self._acausal_trackers:
            self._acausal_trackers[person_id] = AcausalCurveFilter(window_size=5)
        skel, _ = self._acausal_trackers[person_id].process_frame(skel)
        return skel

    @staticmethod
    def _is_skeleton_complete(skel: dict[int, np.ndarray]) -> bool:
        """Mindestanforderungen, damit ein Skelett gerendert wird."""
        torso_pts: int    = sum(1 for k in [5, 6, 11, 12] if k in skel)
        head_limb_pts: int = sum(1 for k in [0,1,2,3,4,7,8,9,10,13,14,15,16] if k in skel)
        return torso_pts >= 2 and head_limb_pts > 0 and len(skel) >= 4

    def _calculate_orientation(
        self,
        skel: dict[int, np.ndarray],
        mem: dict[str, Any],
    ) -> np.ndarray:
        """
        Schätzt den Forward-Vektor der Person per Voting:
          Vote 1 (stärkste): Gesichtspunkte
          Vote 2:            Knie-Beugerichtung
          Vote 3 (schwach):  Bewegungsvektor
        """
        default_forward: np.ndarray = mem.get("forward_vec", np.array([0.0, 0.0, 1.0], dtype=np.float32))

        if not all(k in skel for k in [5, 6, 11, 12]):
            return default_forward

        mid_shoulder: np.ndarray = (skel[5] + skel[6]) / 2.0
        mid_hip: np.ndarray      = (skel[11] + skel[12]) / 2.0

        v_up: np.ndarray   = self._safe_normalize(mid_shoulder - mid_hip, default_forward)
        v_right: np.ndarray = self._safe_normalize(skel[6] - skel[5], default_forward)

        v_forward: np.ndarray = np.cross(v_up, v_right)
        if float(np.linalg.norm(v_forward)) < 1e-3:
            return default_forward
        v_forward /= np.linalg.norm(v_forward)

        vote: float = 0.0

        face_pts: list[int] = [k for k in [0,1,2,3,4] if k in skel]
        if face_pts:
            face_center: np.ndarray = np.mean([skel[k] for k in face_pts], axis=0)
            vote += 2.0 if np.dot(face_center - mid_shoulder, v_forward) > 0 else -2.0

        for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
            if not all(k in skel for k in [hip, knee, ankle]):
                continue
            leg_vec: np.ndarray  = skel[ankle] - skel[hip]
            leg_norm: float = float(np.linalg.norm(leg_vec))
            if leg_norm < 1e-3:
                continue
            leg_dir: np.ndarray  = leg_vec / leg_norm
            proj: float     = float(np.dot(skel[knee] - skel[hip], leg_dir))
            knee_bulge: np.ndarray = skel[knee] - (skel[hip] + proj * leg_dir)
            vote += 1.0 if np.dot(knee_bulge, v_forward) > 0 else -1.0

        curr_com: np.ndarray = (mid_shoulder + mid_hip) / 2.0
        if "last_com" in mem:
            vel: np.ndarray = curr_com - mem["last_com"]
            if float(np.linalg.norm(vel)) > 3.0:
                vote += 0.5 if np.dot(vel, v_forward) > 0 else -0.5
        mem["last_com"] = curr_com

        if vote < 0:
            v_forward = -v_forward

        old_fwd: np.ndarray = mem.get("forward_vec", v_forward)
        smoothed: np.ndarray = 0.15 * v_forward + 0.85 * old_fwd
        smoothed /= np.linalg.norm(smoothed)

        mem["forward_vec"] = smoothed
        return smoothed

    @staticmethod
    def _safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm: float = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-3 else fallback

    @staticmethod
    def _get_or_create(store: dict[int, Any], key: int, cls: type) -> Any:
        if key not in store:
            obj: Any = cls()
            obj.first_seen = time.time()
            store[key] = obj
        return store[key]