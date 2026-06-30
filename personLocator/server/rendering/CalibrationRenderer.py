

import time
from collections import deque
import numpy as np
from typing import Any, Optional, Dict, List, TYPE_CHECKING

from server.config.TrackingConfig import TrackingConfig
from server.core.tracking.Epipolarclusterer import EpipolarClusterer
from server.core.tracking.SkeletonTriangulator import SkeletonTriangulator
from server.core.tracking.Personidtracker import PersonIDTracker
from server.core.tracking.SkeletonPostProcessor import SkeletonPostProcessor
from server.rendering.SceneRenderer import SceneRenderer

if TYPE_CHECKING:
    from server.controllers.ServerController import ServerController
    from server.models.GlobalPerson import GlobalPerson


class CalibrationRenderer(SceneRenderer):
    """
    Haupt-Einstiegspunkt für Kalibrierung, Sensor-Fusion und 3D-Visualisierung.

    Erbt den reinen Render-Layer von SceneRenderer und ergänzt die
    Fusion-Pipeline (Clustern → Triangulieren → Tracken → IK/Smoothing).
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        tracking_mode: str = "hungarian",
        smoothing_mode: str = "one_euro",
        ik_mode: str = "fabrik",
        triangulation_mode: str = "wls",
    ) -> None:
        cfg: TrackingConfig = config or TrackingConfig()

        self.config: TrackingConfig = cfg
        self._clusterer: EpipolarClusterer = EpipolarClusterer(
            max_epipolar_error_cm    = cfg.max_epipolar_error_cm,
            epipolar_ignore_threshold = cfg.epipolar_ignore_threshold,
            color_match_threshold    = cfg.color_match_threshold,
            min_joint_match_ratio    = cfg.min_joint_match_ratio,
        )
        self._triangulator: SkeletonTriangulator = SkeletonTriangulator(triangulation_mode=triangulation_mode)
        self._id_tracker: PersonIDTracker = PersonIDTracker(tracking_mode=tracking_mode)
        self._post_processor: SkeletonPostProcessor = SkeletonPostProcessor(
            config        = cfg,
            smoothing_mode = smoothing_mode,
            ik_mode        = ik_mode,
        )

        self.tracking_mode: str      = tracking_mode
        self.smoothing_mode: str     = smoothing_mode
        self.ik_mode: str            = ik_mode
        self.triangulation_mode: str = triangulation_mode

        self.filter_stats: dict[str, Any] = {
            "ik_resizes": 0,
            "ik_correction_cm": 0.0,
            "kalman_glitches_blocked": 0,
            "kalman_smoothing_cm": 0.0,
            "error_hungarian": 0,
            "error_greedy": 0,
            "id_switches": 0,
            "epipolar_error_sum": 0.0,
            "epipolar_error_count": 0,
            "epipolar_ghosts": 0,
            "reprojection_error": 0.0,
            "health_index": 100.0,
            "server_triangulation_fps": 0.0
        }
        self._fusion_call_times: deque = deque(maxlen=30)

        self.controller: Optional['ServerController'] = None

    def get_filter_stats(self) -> dict[str, Any]:
        """Aggregiert Statistiken aller Subsysteme."""
        self.filter_stats.update(self._clusterer.get_stats())
        self.filter_stats.update(self._id_tracker.get_stats())
        self.filter_stats.update(self._post_processor.get_stats())

        return self.filter_stats

    def get_learned_bones(self, person_id: int) -> dict[str, float]:
        return self._post_processor.get_learned_bones(person_id)

    def _run_fusion(
        self,
        canvas: Optional[np.ndarray],
        persons: list[dict[str, Any]],
        ctx: dict[str, Any],
        zoom: float,
        render_graphics: bool,
        fusion_settings: dict[str, Any],
        room_objects: Optional[list[dict[str, Any]]] = None,
        room_dims: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Vollständige Fusion-Pipeline:
          1. Epipolare Cluster bilden
          2. 3D-Positionen triangulieren
          3. Stabile IDs zuweisen
          4. IK / Smoothing / Orientierung berechnen
          5. Global-Person-Objekte synchronisieren
          6. Optionales Rendering der Fusions-Skelette
        """
        _t_now = time.time()
        self._fusion_call_times.append(_t_now)
        if len(self._fusion_call_times) >= 2:
            _deltas = [self._fusion_call_times[i] - self._fusion_call_times[i-1] for i in range(1, len(self._fusion_call_times))]
            _avg_delta = float(np.mean(_deltas))
            self.filter_stats["server_triangulation_fps"] = 1.0 / _avg_delta if _avg_delta > 0 else 0.0

        clusters: list[dict[str, dict[str, Any]]] = self._clusterer.compute(persons)

        joint_filter: Optional[dict[int, bool]] = fusion_settings.get("joints")
        pre_fusion: list[dict[str, Any]] = self._triangulator.triangulate(clusters, joint_filter)

        repro_errors = [p["reprojection_error"] for p in pre_fusion if "reprojection_error" in p]
        if repro_errors:
            self.filter_stats["reprojection_error"] = float(np.mean(repro_errors))

        if pre_fusion and room_objects:
            ro_id = id(room_objects)
            if getattr(self, "_marker_ro_id", None) != ro_id:
                self._marker_ro_id = ro_id
                _m = [o for o in room_objects if any(s in o.get("name", "").lower()
                      for s in ["marker", "referenz", "boden"])]
                self._marker_pts = np.array([[o["pos_x"], o["pos_y"], o["pos_z"]]
                                             for o in _m], dtype=np.float32) if _m else None
            if self._marker_pts is not None and len(self._marker_pts) > 0:
                loc_errors = []
                for p in pre_fusion:
                    p_pos = np.asarray(p["centroid"], dtype=np.float32)
                    dists = np.linalg.norm(self._marker_pts - p_pos, axis=1)
                    loc_errors.append(float(dists.min()))
                if loc_errors:
                    self.filter_stats["localization_error_cm"] = float(np.mean(loc_errors))
                    self.filter_stats["localization_error_median_cm"] = float(np.median(loc_errors))
                    self.filter_stats["localization_error_rmse_cm"] = float(np.sqrt(np.mean(np.square(loc_errors))))

        self._id_tracker.update(
            pre_fusion,
            cleanup_callbacks=[self._post_processor.cleanup_person],
        )

        now: float = time.time()

        for data in pre_fusion:
            data.setdefault("force_symmetry", bool(fusion_settings.get("force_symmetry", True)))
            self._post_processor.process(data, now)

            self._sync_global_person(data)

            if render_graphics and canvas is not None:
                self.draw_fusion_skeleton(
                    canvas, data, ctx, zoom,
                    to_pt=lambda pt: (int(round(pt[0])), int(round(pt[1]))),
                    show_points=bool(fusion_settings.get("show_points", True)),
                    show_bones=bool(fusion_settings.get("show_bones",  True)),
                    show_arm_extension=bool(fusion_settings.get("show_arm_extension", False)),
                    all_persons=pre_fusion,
                    room_objects=room_objects,
                    room_dims=room_dims,
                )

    def _sync_global_person(self, data: dict[str, Any]) -> None:
        """Spiegelt Fusionsdaten ins entsprechende GlobalPerson-Objekt."""
        if not (self.controller and hasattr(self.controller, "tracker")):
            return

        gp: Optional['GlobalPerson'] = next(
            (p for p in self.controller.tracker.global_persons if p.id == data["id"]),
            None,
        )
        if not gp:
            return

        gp.fusion_skel  = data.get("skel", {})
        gp.forward_vec  = data.get("forward_vec")
        gp.current_pointers = data.get("pointing", {})

        self._update_person_body_dimensions(gp, int(data["id"]))

    def _update_person_body_dimensions(self, gp: Any, person_id: int) -> None:
        """Berechnet Körperhöhe und -breite aus den gelernten Knochen."""
        bones: dict[str, float] = self._post_processor.get_learned_bones(person_id)
        if not bones:
            return

        leg: float = max(
            bones.get("leg_l_up", 45.0) + bones.get("leg_l_down", 42.0),
            bones.get("leg_r_up", 45.0) + bones.get("leg_r_down", 42.0),
        )
        spine: float = 55.0
        skel: dict[int, np.ndarray] = getattr(gp, "fusion_skel", {})
        if 5 in skel and 11 in skel:
            spine = float(np.linalg.norm(skel[5] - skel[11]))

        gp.fused_height = float(leg + spine + 28.0)

        if "arm_l_up" in bones and "arm_r_up" in bones and 5 in skel and 6 in skel:
            gp.fused_width = float(np.linalg.norm(skel[5] - skel[6]))
        elif 5 in skel and 6 in skel:
            gp.fused_width = 45.0

    def render_3d_scene(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """
        Wie SceneRenderer.render_3d_scene, aber mit injiziertem Fusion-Callback.
        Alle Parameter werden 1:1 durchgereicht.
        """
        kwargs.setdefault("on_master_fusion", self._run_fusion)
        return super().render_3d_scene(*args, **kwargs)