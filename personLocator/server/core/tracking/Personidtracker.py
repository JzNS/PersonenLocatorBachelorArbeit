import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, List, Dict, Set, Optional, Callable

from server.rendering.DrawingUtils import DrawingUtils


class PersonIDTracker:
    """
    Weist triangulierten Personen stabile IDs zu (DeepSort-inspiriert).

    Unterstützt zwei Matching-Modi:
      "hungarian" — globales Optimum (Kuhn-Munkres), robust bei Verdeckungen
      "greedy"    — schnelles gieriges Matching, fehleranfälliger

    Reinigt veraltete Tracks automatisch nach `max_track_age` Sekunden.
    """

    STRICT_RADIUS: float   = 80.0
    MAX_TRACK_AGE: float   = 3.0

    def __init__(self, tracking_mode: str = "hungarian") -> None:
        self.tracking_mode: str = tracking_mode

        self._track_memory: dict[int, dict[str, Any]] = {}
        self._id_counter: int = 1

        self._stats_error_hungarian: int = 0
        self._stats_error_greedy: int    = 0
        self._stats_id_switches: int     = 0
        self.last_global_state: dict[int, dict[str, Any]] = {}

    def update(
        self,
        pre_fusion_data: list[dict[str, Any]],
        cleanup_callbacks: Optional[list[Callable[[int], None]]] = None,
    ) -> None:
        """
        Weist jedem Eintrag in `pre_fusion_data` eine stabile `id` zu.

        `cleanup_callbacks`: Liste von Funktionen, die mit einer toten Track-ID
        aufgerufen werden (z. B. um Filter-Objekte zu löschen).
        """
        now: float = time.time()
        self._cleanup_dead_tracks(now, cleanup_callbacks or [])

        old_ids: list[int]    = list(self._track_memory.keys())
        old_states: list[dict[str, Any]] = list(self._track_memory.values())

        if not old_ids or not pre_fusion_data:
            for data in pre_fusion_data:
                self._register_new_track(data, now)
            return

        cost_matrix: np.ndarray = self._build_cost_matrix(old_states, pre_fusion_data, now)
        used_new: set[int]    = self._run_matching(old_ids, pre_fusion_data, cost_matrix, now)

        for j, data in enumerate(pre_fusion_data):
            if j not in used_new:
                self._register_new_track(data, now)

        self.last_global_state = {
            pid: {"pos": val["pos"], "colors": val["colors"]}
            for pid, val in self._track_memory.items()
        }

    @property
    def track_memory(self) -> dict[int, dict[str, Any]]:
        """Nur-Lesen-Zugriff auf den Track-Speicher (z. B. für den Renderer)."""
        return self._track_memory

    def get_stats(self) -> dict[str, int]:
        return {
            "error_hungarian": self._stats_error_hungarian,
            "error_greedy":    self._stats_error_greedy,
            "id_switches":     self._stats_id_switches,
        }

    def _cleanup_dead_tracks(
        self,
        now: float,
        callbacks: list[Callable[[int], None]],
    ) -> None:
        dead_keys: list[int] = [k for k, v in self._track_memory.items()
                     if now - v["last_seen"] > self.MAX_TRACK_AGE]
        for k in dead_keys:
            del self._track_memory[k]
            for cb in callbacks:
                cb(k)

    def _register_new_track(self, data: dict[str, Any], now: float) -> None:
        data["id"] = self._id_counter
        self._track_memory[self._id_counter] = {
            "pos":       data["centroid"],
            "colors":    data["colors"],
            "last_seen": now,
        }
        self._id_counter += 1

    def _build_cost_matrix(
        self,
        old_states: list[dict[str, Any]],
        pre_fusion_data: list[dict[str, Any]],
        now: float,
    ) -> np.ndarray:
        """Baut die Kostenmatrix auf (Distanz + Farbe, gewichtet nach Kontext)."""
        n_old: int = len(old_states)
        n_new: int = len(pre_fusion_data)
        cost: np.ndarray  = np.full((n_old, n_new), 1000.0)

        for i, o_state in enumerate(old_states):
            time_lost: float = now - o_state["last_seen"]
            in_group: bool  = any(
                float(np.linalg.norm(o_state["pos"] - other["pos"])) < 100.0
                for other in old_states if other is not o_state
            )

            for j, data in enumerate(pre_fusion_data):
                dist: float = float(np.linalg.norm(o_state["pos"] - data["centroid"]))
                if dist > self.STRICT_RADIUS:
                    continue

                dist_norm: float = dist / self.STRICT_RADIUS
                col_cost: float  = DrawingUtils.calculate_color_distance(o_state["colors"], data["colors"])

                if in_group and time_lost > 0.1:
                    if col_cost > 0.30:
                        continue
                    cost[i, j] = 0.1 * dist_norm + 0.9 * col_cost
                else:
                    cost[i, j] = 0.7 * dist_norm + 0.3 * col_cost

        return cost

    def _run_matching(
        self,
        old_ids: list[int],
        pre_fusion_data: list[dict[str, Any]],
        cost_matrix: np.ndarray,
        now: float,
    ) -> set[int]:
        """Führt Matching durch und gibt die Menge genutzter neuer Indizes zurück."""
        if self.tracking_mode == "greedy":
            return self._match_greedy(old_ids, pre_fusion_data, cost_matrix, now)
        else:
            return self._match_hungarian(old_ids, pre_fusion_data, cost_matrix, now)

    def _match_hungarian(
        self,
        old_ids: list[int],
        pre_fusion_data: list[dict[str, Any]],
        cost_matrix: np.ndarray,
        now: float,
    ) -> set[int]:
        used: set[int] = set()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 10.0:
                continue
            if cost_matrix[r, c] > 0.8:
                self._stats_error_hungarian += 1
            self._assign_id(old_ids[r], pre_fusion_data[c], now)
            used.add(c)

        return used

    def _match_greedy(
        self,
        old_ids: list[int],
        pre_fusion_data: list[dict[str, Any]],
        cost_matrix: np.ndarray,
        now: float,
    ) -> set[int]:
        flat: list[tuple[float, int, int]] = [
            (float(cost_matrix[r, c]), int(r), int(c))
            for r in range(len(old_ids))
            for c in range(len(pre_fusion_data))
        ]
        flat.sort(key=lambda x: x[0])
        used_r: set[int] = set()
        used_c: set[int] = set()

        for cost, r, c in flat:
            if cost >= 10.0 or r in used_r or c in used_c:
                continue
            if cost > 0.5:
                self._stats_error_greedy += 1
            self._assign_id(old_ids[r], pre_fusion_data[c], now)
            used_r.add(r)
            used_c.add(c)

        return used_c

    def _assign_id(
        self,
        assigned_id: int,
        data: dict[str, Any],
        now: float,
    ) -> None:
        # ID-Switch Erkennung: Wenn die Farbe sich massiv ändert (> 0.4)
        if data["colors"] and self._track_memory[assigned_id]["colors"]:
            col_diff = DrawingUtils.calculate_color_distance(self._track_memory[assigned_id]["colors"], data["colors"])
            if col_diff > 0.4:
                self._stats_id_switches += 1

        data["id"] = assigned_id
        self._track_memory[assigned_id].update({
            "pos":       data["centroid"],
            "last_seen": now,
        })
        if data["colors"]:
            self._track_memory[assigned_id]["colors"] = data["colors"]