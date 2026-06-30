from server.core.logger import get_logger
import numpy as np
import copy
import collections
import time
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from psycopg2.extras import RealDictCursor
from server.models.GlobalPerson import GlobalPerson
from server.controllers.ServerPersonTracker import ServerPersonTracker
from server.core.database.PersonDatabase import PersonDatabase
from server.network.ServerCommandHandler import ServerCommandHandler
from server.network.ServerConnector import ServerConnector
from server.network.ServerBeacon import ServerBeacon
from server.network.ServerCommandSender import ServerCommandSender
from server.core.database.SystemDatabase import SystemDatabase

logger = get_logger("server.controller")


if TYPE_CHECKING:
    from server.gui.ServerDashboard import ServerDashboard


class ServerController:
    """
    Zentrale Steuereinheit (Mediator).
    Verbindet Netzwerk, Tracker und Dashboard.
    """

    _KP_NAMES: dict[int, str] = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    }

    def _log_skeleton_to_file(self, camera_name: str, persons: list, source: str = "client") -> None:
        """Schreibt Bbox/Keypoint/Farbdaten als JSON-Lines in skeleton_log.jsonl.
        source="client"  → 2-D Detektionen mit Bbox, Keypoints und Joint-Farben.
        source="server_fusion" → 3-D Fusionsskelett mit GlobalPerson-Farbprofil.
        """
        import os, json, math

        now = __import__("time").time()
        log_dir = getattr(self, "_log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        def _safe_float(v):
            try:
                f = float(v)
                return None if (math.isnan(f) or math.isinf(f)) else round(f, 3)
            except Exception:
                return None

        def _coords3(arr):
            if arr is None or not hasattr(arr, "__len__") or len(arr) < 3:
                return None
            vals = [_safe_float(arr[i]) for i in range(3)]
            return vals if all(v is not None for v in vals) else None

        entries = []

        if source == "client":
            for i, p in enumerate(persons):
                entry = {
                    "ts": round(now, 6),
                    "frame_id": self._frame_counter,
                    "test_block": self.current_test_block,
                    "source": "client",
                    "camera": camera_name,
                    "person_idx": i,
                    "person_id": str(p.get("id", i)),
                }
                bbox = p.get("bbox", [])
                if bbox:
                    entry["bbox"] = [round(float(v), 1) for v in bbox[:4]]
                    entry["bbox_conf"] = _safe_float(p.get("bbox_confidence", p.get("confidence", 0)))
                kps_raw = p.get("keypoints", [])
                metrics = p.get("metrics", {})
                joint_colors: dict = metrics.get("joint_colors", {})
                stable_colors: dict = metrics.get("stable_colors", {})
                kp_list = []
                for kp in kps_raw:
                    if not isinstance(kp, dict):
                        continue
                    kid = int(kp.get("id", -1))
                    x = _safe_float(kp.get("x", kp.get("px", 0)))
                    y = _safe_float(kp.get("y", kp.get("py", 0)))
                    conf = _safe_float(kp.get("c", kp.get("conf", kp.get("confidence", 0))))
                    kp_entry = {
                        "id": kid,
                        "name": self._KP_NAMES.get(kid, f"kp_{kid}"),
                        "x": x, "y": y, "conf": conf,
                    }
                    col = joint_colors.get(kid, kp.get("color"))
                    if col is not None:
                        kp_entry["color"] = col
                    kp_list.append(kp_entry)
                if kp_list:
                    entry["keypoints_2d"] = kp_list
                if stable_colors:
                    entry["stable_colors"] = {str(k): v for k, v in stable_colors.items()}
                pos2 = p.get("pos")
                if pos2 is not None and hasattr(pos2, "__len__") and len(pos2) >= 2:
                    c2 = _coords3(pos2)
                    if c2:
                        entry["pos_projected"] = c2
                entries.append(entry)

        elif source == "server_fusion":
            for gp in persons:
                entry = {
                    "ts": round(now, 6),
                    "frame_id": self._frame_counter,
                    "test_block": self.current_test_block,
                    "source": "server_fusion",
                    "camera": "FUSION",
                    "track_id": str(getattr(gp, "id", "")),
                    "person_name": str(getattr(gp, "name", "")),
                }
                c3 = _coords3(getattr(gp, "pos", None))
                if c3:
                    entry["centroid"] = c3
                skel = getattr(gp, "fusion_skel", {})
                skel_3d = {}
                for kid, coords in (skel.items() if isinstance(skel, dict) else []):
                    kid_int = int(kid)
                    c = _coords3(coords)
                    if c:
                        skel_3d[self._KP_NAMES.get(kid_int, f"kp_{kid_int}")] = c
                if skel_3d:
                    entry["skel_3d"] = skel_3d
                cp = getattr(gp, "color_profile", {})
                if cp:
                    serialized_cp = {}
                    for orientation, kp_colors in cp.items():
                        if isinstance(kp_colors, dict) and kp_colors:
                            serialized_cp[orientation] = {
                                self._KP_NAMES.get(int(k), f"kp_{k}"): [
                                    round(float(v), 3) for v in (vals if hasattr(vals, "__len__") else [vals])
                                ]
                                for k, vals in kp_colors.items()
                                if vals is not None
                            }
                    if serialized_cp:
                        entry["color_profile"] = serialized_cp
                fv = _coords3(getattr(gp, "forward_vec", None))
                if fv:
                    entry["forward_vec"] = fv
                entries.append(entry)

        if not entries:
            return

        def _write_skel(filepath):
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    for e in entries:
                        f.write(json.dumps(e, default=str) + "\n")
            except Exception as ex:
                logger.error(f"Fehler beim Schreiben von {filepath}: {ex}")

        block_slug = self.current_test_block.replace(" ", "_")
        _write_skel(os.path.join(log_dir, f"skeleton_log_{block_slug}.jsonl"))
        if self._session_log_file:
            _write_skel(self._session_log_file.replace(".csv", "_skeleton.jsonl"))

    def __init__(self, port: int, dashboard: Optional['ServerDashboard'] = None) -> None:
        self.port: int = port
        self.dashboard: Optional['ServerDashboard'] = dashboard
        self.view_master: Optional[Any] = None
        
        if self.dashboard:
            self.dashboard.set_controller(self)

        self.person_db: PersonDatabase = PersonDatabase()
        self.system_db: SystemDatabase = SystemDatabase()
        self.config_cache: Dict[str, Any] = {}
        self.room_objects: list[dict[str, Any]] = []

        self.tracker: ServerPersonTracker = ServerPersonTracker(
            person_database=self.person_db,
            person_class_ref=GlobalPerson
        )

        self.connector: ServerConnector = ServerConnector(port=self.port)
        self.beacon: ServerBeacon = ServerBeacon(port=self.port)
        self.sender: ServerCommandSender = ServerCommandSender(self.connector)

        self.handler: ServerCommandHandler = ServerCommandHandler(
            server_connector=self.connector,
            command_sender=self.sender,
            controller=self
        )
        self.connector.set_command_handler(self.handler)
        self.refresh_config_cache()
        self.reload_room_objects()

        self.current_test_block: str = "Block_1_Inferenzlatenz"
        self._frame_counter: int = 0
        self._recording_active: bool = False
        self._recording_frame_limit: int = 0
        self._recording_time_limit: float = 0.0
        self._recording_start_time: float = time.time()
        self._recording_session_frames: int = 0
        self._stats_prev: dict = {}
        self._log_dir: str = "logs"
        self._session_log_file: str = ""
        self._cam_net_samples: dict = {}    # Uhrdrift-Korrektur pro Kamera
        self._cam_clock_offset: dict = {}   # letzter Offset-Schätzer pro Kamera
        self._cam_offset_counter: dict = {} # Frame-Zaehler fuer Offset-Update

    def reload_room_objects(self) -> None:
        """Lädt die im DB gespeicherten Raum-Objekte neu in den Cache."""
        try:
            self.room_objects = self.system_db.get_all_room_objects()
        except Exception as e:
            self.room_objects = []
            logger.error(f"Fehler beim Laden der Raum-Objekte: {e}")

    def start(self) -> None:
        self.connector.start()
        self.beacon.start()
        if self.dashboard:
            self.dashboard.log_message(f"Server gestartet auf Port {self.port}")

    def refresh_config_cache(self) -> None:
        """Lädt den Cache neu und triggert GUI-Updates."""
        try:
            new_cache = {}

            all_cams = self.system_db.get_all_cameras()
            for cam_name in all_cams:
                settings = self.system_db.get_camera_settings(cam_name)
                if settings:
                    new_cache[cam_name] = settings

            if "Camera_ALL" not in new_cache:
                new_cache["Camera_ALL"] = {}
            new_cache["Camera_ALL"]["lens_profiles"] = self.system_db.get_all_lens_profiles()

            self.config_cache = new_cache

            for cam_name in all_cams:
                self._fetch_camera_data_from_db(cam_name)
            
            self.reload_room_objects()
            
            self.log_to_dashboard("🔄 System-Cache (inkl. 3D-Matrizen) wurde erfolgreich aktualisiert.")

            if self.dashboard:
                self.dashboard.sig_update_camera_view.emit("SYSTEM_REFRESH", [])

            if hasattr(self, 'db_viewer') and getattr(self, 'db_viewer') and getattr(self, 'db_viewer').isVisible():
                getattr(self, 'db_viewer').external_data_updated()

        except Exception as e:
            import traceback
            logger.error(f"Fehler beim Cache-Refresh: {e}")
            traceback.print_exc()

    def execute_client_handshake(self, client_name: str) -> None:
        """Handshakes sind die magischen Momente, in denen ein Client zum ersten Mal mit dem Server spricht.
        Hier entscheiden wir, ob er die volle 3D-Konfiguration oder die Lite-Version bekommt."""
        if not hasattr(self, 'system_db'):
            return
        try:
            self.system_db.register_camera_if_not_exists(client_name)
            settings: dict[str, Any] = self.system_db.get_camera_settings(client_name)

            if not settings:
                logger.warning(f"Konnte Settings für {client_name} nicht laden. Sende Lite-Config.")
                self.sender.send_message(client_name, "CONFIG_LITE", {})
                return

            perf_mode: bool = settings.get("performance_mode", False)

            if perf_mode:
                self.log_to_dashboard(f"Handshake: {client_name} ist im Performance-Modus.")
                payload: dict[str, Any] = {
                    client_name: settings,
                    "Camera_ALL": {
                        "lens_profiles": self.system_db.get_all_lens_profiles()
                    }
                }
                self.sender.send_message(client_name, "CONFIG_LITE", payload)
            else:
                self.log_to_dashboard(f"Handshake: {client_name} ist im 3D-Modus.")
                self.send_full_config_to_client(client_name)

        except Exception as e:
            self.log_to_dashboard(f"Fehler beim Handshake mit {client_name}: {e}")

    def sync_all_clients(self) -> None:
        """Sobald sich die DB ändert, müssen alle Clients ihre Konfiguration aktualisieren."""
        if not hasattr(self, 'connector') or not self.connector.clients:
            return

        self.log_to_dashboard(f"🔄 Live-Sync: Sende DB-Update an {len(self.connector.clients)} verbundene Clients...")
        for client_name in list(self.connector.clients.keys()):
            self.execute_client_handshake(client_name)

    def send_full_config_to_client(self, client_name: str) -> None:
        """Sendet die vollständige 3D-Konfiguration an den Client."""
        try:
            settings: dict[str, Any] = self.system_db.get_camera_settings(client_name)
            global_rects: list[dict[str, Any]] = self.system_db.get_all_world_rectangles()
            pixels: dict[str, Any] = self.system_db.get_camera_pixels(client_name)

            payload: dict[str, Any] = {
                client_name: settings,
                "Camera_ALL": {
                    "reference_rectangles": copy.deepcopy(global_rects),
                    "room_dimensions": self.system_db.get_global_room()
                }
            }
            payload[client_name]["rectangle_pixels"] = pixels
            payload["Camera_ALL"]["lens_profiles"] = self.system_db.get_all_lens_profiles()

            self.sender.send_message(client_name, "CONFIG_FULL", payload)
        except Exception as e:
            self.log_to_dashboard(f"Fehler CONFIG_FULL für {client_name}: {e}")

    def get_unmerged_tracks(self) -> Dict[str, list[dict[str, Any]]]:
        return self.tracker.get_raw_tracks()

    def execute_manual_merge(self, name: str, selection_list: List[dict[str, Any]]) -> None:
        msg: str = self.tracker.force_merge_and_calibrate(name, selection_list)
        self.log_to_dashboard(msg)

    def update_camera_view_logic(self, camera_name: str, person_list: list[dict[str, Any]], t3: float = 0.0, t2: float = 0.0, t1: float = 0.0, inf_time: float = 0.0, client_fps: float = 0.0) -> None:
        from server.core.math.GeometryMath import GeometryMath
        import time

        if camera_name not in self.config_cache:
            self._fetch_camera_data_from_db(camera_name)

        cam_conf: dict[str, Any] = self.config_cache.get(camera_name, {})
        pixel_points: list[Any] = cam_conf.get("pixel_points", [])
        world_points_3d: list[np.ndarray] = cam_conf.get("world_points_3d", [])
        world_rects: list[dict[str, Any]] = cam_conf.get("custom_rectangles", [])
        res: tuple[int, int] = cam_conf.get("resolution", (1920, 1080))
        cam_matrix: Optional[np.ndarray] = cam_conf.get("camera_matrix")
        dist_c: Optional[np.ndarray] = cam_conf.get("dist_coeffs")
        camera_type: str = str(cam_conf.get("camera_type", "Standard"))

        if t1 == 0 and person_list:
            t1 = person_list[0].get("t1", 0.0)

        if inf_time == 0 and person_list:
            inf_time = person_list[0].get("inference_time_ms", 0.0)

        if t1 == 0:
            logger.debug(f"⚠️ Latency Trace [{camera_name}]: t1 is 0.0! (t2={t2}, t3={t3})")

        clean_persons: list[dict[str, Any]] = []
        for p in person_list:
            bbox: list[float] = p.get('bbox', [])
            kps: list[dict[str, Any]] = p.get('keypoints', [])
            calc_pos: Optional[np.ndarray] = None

            if len(pixel_points) >= 4 and len(world_points_3d) >= 4 and len(bbox) == 4:
                calc_pos = GeometryMath.smart_project_position(
                    kps, bbox, pixel_points, world_points_3d, 180.0, world_rects,
                    img_size=res, camera_matrix_override=cam_matrix, dist_coeffs_override=dist_c,
                    camera_name=camera_name
                )

            p['pos'] = calc_pos if calc_pos is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p['camera_type'] = camera_type
            clean_persons.append(p)

        self.tracker.update_camera_data(camera_name, clean_persons)
        self._log_skeleton_to_file(camera_name, clean_persons, source="client")
        
        self._frame_counter += 1

        # t4: Pipeline Exit (nach Fusion und Tracking)
        t4 = time.time()

        if camera_name not in self._cam_net_samples:
            self._cam_net_samples[camera_name] = collections.deque(maxlen=120)
            self._cam_clock_offset[camera_name] = -0.0005
            self._cam_offset_counter[camera_name] = 0
        _apparent_net_s = (t3 - t2) if (t3 > 0 and t2 > 0) else 0.0
        self._cam_net_samples[camera_name].append(_apparent_net_s)
        self._cam_offset_counter[camera_name] += 1
        if self._cam_offset_counter[camera_name] >= 30:
            self._cam_offset_counter[camera_name] = 0
            _samp = sorted(self._cam_net_samples[camera_name])
            _p05  = _samp[max(0, len(_samp) // 20)]
            self._cam_clock_offset[camera_name] = _p05 - 0.0005
        _clock_offset_s  = self._cam_clock_offset[camera_name]
        _corr_net_ms     = max((_apparent_net_s - _clock_offset_s) * 1000.0, 0.3)
        _client_ms       = (t2 - t1) * 1000.0 if (t2 > 0 and t1 > 0) else 0.0
        _srv_ms          = (t4 - t3) * 1000.0 if (t4 > 0 and t3 > 0) else 0.0
        _corr_e2e_ms     = _client_ms + _corr_net_ms + _srv_ms
        latencies = {
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "inference_time_ms": float(inf_time) if inf_time else 0.0,
            "client_latency_ms": _client_ms,
            "network_latency_ms": _corr_net_ms,
            "server_latency_ms": _srv_ms,
            "end_to_end_latency_ms": _corr_e2e_ms,
            "clock_offset_ms": _clock_offset_s * 1000.0,
        }

        if hasattr(self, 'view_master') and self.view_master:
            if "cam_latencies" not in self.view_master.renderer.filter_stats:
                self.view_master.renderer.filter_stats["cam_latencies"] = {}

            latencies["client_fps"] = client_fps
            self.view_master.renderer.filter_stats["cam_latencies"][camera_name] = latencies

            self.view_master.renderer.filter_stats.update(latencies)

            self._log_latency_to_file(camera_name, latencies)

        if self.dashboard:
            self.dashboard.update_camera_data(camera_name, clean_persons)

            export_list: list[dict[str, Any]] = []
            for gp in self.tracker.global_persons:
                active_cams: list[str] = list(gp.client_observations.keys())
                cam_info: str = " + ".join(active_cams) if active_cams else "Unbekannt"
                calc_mode: str = "Triangulation" if len(active_cams) >= 2 or "Triangulation" in cam_info else "Raycasting"

                export_list.append({
                    "id": gp.id,
                    "pos": gp.pos,
                    "height": gp.height,
                    "status": gp.name,
                    "keypoints": gp.keypoints,
                    "skeleton_3d": gp.skeleton_3d,
                    "mode": calc_mode,
                    "cam_info": cam_info,
                    "cameras": active_cams
                })

            self.dashboard.update_camera_data("MASTER_FUSION", export_list)


    def set_log_dir(self, path: str) -> None:
        """Setzt das Ausgabeverzeichnis für eval_log.csv und Session-Dateien."""
        import os
        self._log_dir = path
        os.makedirs(path, exist_ok=True)
        self.log_to_dashboard(f"📁 Log-Verzeichnis: {path}")

    def set_test_block(self, block: str) -> None:
        """Wechselt den aktiven Test-Block-Tag für das CSV-Log."""
        self.current_test_block = block
        self.log_to_dashboard(f"📊 Test-Block: {block}")

    def start_recording(self, frame_limit: int = 0, time_limit: float = 0.0) -> str:
        """Startet isolierte Session-Datei (eval_session_TIMESTAMP.csv).
        Das Haupt-Log eval_log.csv laeuft immer weiter."""
        import os
        self._recording_active = True
        self._recording_frame_limit = frame_limit
        self._recording_time_limit = time_limit
        self._recording_start_time = time.time()
        self._recording_session_frames = 0
        self._stats_prev = {}
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = self._log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._session_log_file = os.path.join(log_dir, f"eval_session_{ts}.csv")
        limit_str = f"Frames: {frame_limit}" if frame_limit > 0 else f"Zeit: {time_limit}s" if time_limit > 0 else "unbegrenzt"
        self.log_to_dashboard(f"🔴 REC → {self._session_log_file} ({limit_str})")
        return self._session_log_file

    def stop_recording(self) -> None:
        """Beendet die aktive Session-Datei. Haupt-Log laeuft weiter."""
        self._recording_active = False
        session_file = self._session_log_file or ""
        self._session_log_file = None
        self.log_to_dashboard(f"⬛ REC gestoppt — {self._recording_session_frames} Frames in {session_file}")

    def set_filter_mode(self, mode: str) -> None:
        """Wechselt den Rauschfilter (none/one_euro/kalman) zur Laufzeit."""
        if hasattr(self, 'view_master') and self.view_master:
            self.view_master.renderer._post_processor.smoothing_mode = mode
            self.view_master.renderer.smoothing_mode = mode
        self.log_to_dashboard(f"🔄 Filter-Modus: {mode}")

    def set_triangulation_mode(self, mode: str) -> None:
        """Wechselt den Triangulations-Algorithmus (wls/lm) zur Laufzeit."""
        if hasattr(self, 'view_master') and self.view_master:
            self.view_master.renderer._triangulator.triangulation_mode = mode
            self.view_master.renderer.triangulation_mode = mode
        self.log_to_dashboard(f"🔄 Triangulation: {mode}")

    def set_tracking_mode(self, mode: str) -> None:
        """Wechselt den ID-Tracking-Algorithmus (hungarian/greedy) zur Laufzeit."""
        if hasattr(self, 'view_master') and self.view_master:
            self.view_master.renderer._id_tracker.tracking_mode = mode
            self.view_master.renderer.tracking_mode = mode
        self.log_to_dashboard(f"🔄 Tracking: {mode}")

    def _log_latency_to_file(self, camera_name: str, latencies: dict[str, float]) -> None:
        """Schreibt einen vollstaendigen Frame-Record in eval_log.csv.

        Alle kumulativen Metriken werden als Delta (Aenderung gegenueber dem letzten Frame)
        gespeichert, damit jede Zeile ohne Vorkenntnisse direkt auswertbar ist.
        """
        import os
        import csv
        import math

        now = time.time()

        if self._recording_active:
            if self._recording_frame_limit > 0 and self._recording_session_frames >= self._recording_frame_limit:
                self.stop_recording()
            elif self._recording_time_limit > 0.0 and (now - self._recording_start_time) >= self._recording_time_limit:
                self.stop_recording()
            else:
                self._recording_session_frames += 1

        log_dir = getattr(self, "_log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)


        stats: dict = {}
        smoothing_mode = triangulation_mode = tracking_mode = ik_mode = "unknown"
        if hasattr(self, 'view_master') and self.view_master:
            r = self.view_master.renderer
            stats = r.get_filter_stats()
            smoothing_mode     = getattr(r, 'smoothing_mode',     'unknown')
            triangulation_mode = getattr(r, 'triangulation_mode', 'unknown')
            tracking_mode      = getattr(r, 'tracking_mode',      'unknown')
            ik_mode            = getattr(r._post_processor, 'ik_mode', 'unknown') if hasattr(r, '_post_processor') else 'unknown'

        CUMULATIVE_KEYS = [
            "id_switches",
            "error_hungarian",
            "error_greedy",
            "epipolar_ghosts",
            "ik_resizes",
            "ik_correction_cm",
            "kalman_glitches_blocked",
            "kalman_smoothing_cm",
        ]

        def _delta(key: str) -> float:
            cur = float(stats.get(key, 0))
            prev = float(self._stats_prev.get(key, cur))
            d = max(0.0, cur - prev)
            self._stats_prev[key] = cur
            return d

        deltas = {k: _delta(k) for k in CUMULATIVE_KEYS}

        pos_x = pos_y = pos_z = ""
        track_id = ""
        try:
            active = [gp for gp in self.tracker.global_persons if now - gp.last_update <= 2.0]
            if active:
                gp = active[0]
                if gp.pos is not None and len(gp.pos) >= 3:
                    px, py, pz = float(gp.pos[0]), float(gp.pos[1]), float(gp.pos[2])
                    if not any(math.isnan(v) or math.isinf(v) for v in [px, py, pz]):
                        pos_x, pos_y, pos_z = round(px, 2), round(py, 2), round(pz, 2)
                track_id = gp.id
        except Exception:
            pass

        # Fusion-Skelett aller aktiven Personen ins skeleton_log schreiben
        if active:
            self._log_skeleton_to_file("FUSION", active, source="server_fusion")

        def _f(v, dec: int = 2) -> str:
            try:
                f = float(v)
                return "" if math.isnan(f) or math.isinf(f) else str(round(f, dec))
            except Exception:
                return ""

        CSV_HEADER = [
            "system_time", "frame_id", "test_block", "camera",
            "smoothing_mode", "triangulation_mode", "tracking_mode", "ik_mode",
            "t1", "t2", "t3", "t4",
            "inference_ms", "network_ms", "server_ms", "e2e_ms",
            "client_fps", "server_triangulation_fps",
            "pos_x", "pos_y", "pos_z", "track_id",
            "repro_error_px", "epipolar_error_avg", "loc_error_cm", "loc_rmse_cm",
            "epipolar_ghosts_delta",
            "id_switches_delta", "error_hungarian_delta", "error_greedy_delta",
            "ik_resizes_delta", "ik_correction_cm_delta",
            "kalman_blocked_delta", "kalman_smoothing_cm_delta",
            "health_index", "camera_count", "clock_offset_ms",
        ]

        _e2e_raw  = float(latencies.get("end_to_end_latency_ms", 0) or 0)
        _infer_ms = float(latencies.get("inference_time_ms", 0) or 0)
        _net_ms   = float(latencies.get("network_latency_ms", 0) or 0)
        _srv_ms   = float(latencies.get("server_latency_ms", 0) or 0)
        _e2e_ms   = _e2e_raw if _e2e_raw >= 0 else _infer_ms + max(_net_ms, 0.0) + _srv_ms
        camera_count = len(self.connector.clients) if hasattr(self, 'connector') and hasattr(self.connector, 'clients') else 0

        row = [
                    round(now, 6),
                    self._frame_counter,
                    self.current_test_block,
                    camera_name,
                    smoothing_mode,
                    triangulation_mode,
                    tracking_mode,
                    ik_mode,
                    _f(latencies.get("t1",  float("nan")), 6),
                    _f(latencies.get("t2",  float("nan")), 6),
                    _f(latencies.get("t3",  float("nan")), 6),
                    _f(latencies.get("t4",  float("nan")), 6),
                    _f(latencies.get("inference_time_ms", 0), 2),
                    _f(latencies.get("network_latency_ms", 0), 2),
                    _f(latencies.get("server_latency_ms", 0), 2),
                    _f(_e2e_ms, 2),
                    _f(latencies.get("client_fps", 0), 1),
                    _f(stats.get("server_triangulation_fps", 0), 1),
                    pos_x, pos_y, pos_z, track_id,
                    _f(stats.get("reprojection_error", 0), 3),
                    _f(stats.get("epipolar_error_avg", 0), 3),
                    _f(stats.get("localization_error_cm", 0), 2),
                    _f(stats.get("localization_error_rmse_cm", 0), 2),
                    _f(deltas["epipolar_ghosts"], 0),
                    _f(deltas["id_switches"], 0),
                    _f(deltas["error_hungarian"], 0),
                    _f(deltas["error_greedy"], 0),
                    _f(deltas["ik_resizes"], 0),
                    _f(deltas["ik_correction_cm"], 3),
                    _f(deltas["kalman_glitches_blocked"], 0),
                    _f(deltas["kalman_smoothing_cm"], 3),
            _f(stats.get("health_index", 100.0), 1),
            camera_count,
            _f(latencies.get("clock_offset_ms", 0.0), 2),
        ]

        def _write_to(filepath):
            new_file = not os.path.isfile(filepath)
            try:
                with open(filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if new_file:
                        writer.writerow(CSV_HEADER)
                    writer.writerow(row)
            except Exception as e:
                logger.error(f"Fehler beim Schreiben von {filepath}: {e}")

        block_slug = self.current_test_block.replace(" ", "_")
        _write_to(os.path.join(log_dir, f"eval_log_{block_slug}.csv"))
        if self._session_log_file:
            _write_to(self._session_log_file)

    def _fetch_camera_data_from_db(self, camera_name: str) -> None:
        """Holt die Daten EXAKT EINMAL aus der DB, parst das Linsenprofil und legt die 3D-Pose im RAM-Cache ab."""
        pixel_points: list[list[float]] = []
        world_points_3d: list[list[float]] = []
        world_rects: list[dict[str, Any]] = []
        room_dims: dict[str, float] = {"width": 600.0, "height": 250.0, "depth": 800.0}
        res: list[int] = [1920, 1080]
        cam_matrix: Optional[Union[list[list[float]], np.ndarray]] = None
        dist_c: Optional[Union[list[float], np.ndarray]] = None
        active_prof: str = "default"

        if hasattr(self, 'system_db') and self.system_db.db_pool:
            conn = self.system_db.db_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                    r = cursor.fetchone()
                    if r:
                        room_dims = {"width": float(r["width_cm"]), "height": float(r["height_cm"]), "depth": float(r["depth_cm"])}

                    cursor.execute("""
                        SELECT w.id, w.display_id, w.type, w.is_active AS global_active, cr.is_active AS local_active
                        FROM world_rectangles w
                        LEFT JOIN camera_rectangles cr ON w.id = cr.rect_id AND cr.camera_name = %s
                    """, (camera_name,))

                    for rect in cursor.fetchall():
                        r_id: str = str(rect["id"])

                        # Lokaler Status (Checkbox der Kamera) schlägt globalen Status
                        actual_active: bool = bool(rect["local_active"]) if rect["local_active"] is not None else bool(rect["global_active"])

                        # Der Hauptraum ist für die Optik immer an
                        if r_id == "MAIN_ROOM_CALIB" or rect["type"] == "Haupt-Kalibrierung":
                            actual_active = True

                        cursor.execute("""
                            SELECT c.label, c.x, c.y, c.z, m.px, m.py
                            FROM rectangle_corners_3d c
                            LEFT JOIN camera_pixel_mapping m ON m.corner_3d_id = c.id AND m.camera_name = %s
                            WHERE c.rect_id = %s ORDER BY c.id ASC
                        """, (camera_name, r_id))

                        corners = cursor.fetchall()
                        rect_data: dict[str, Any] = {"internal_id": r_id, "display_id": rect["display_id"], "type": rect["type"],
                                     "is_active": actual_active, "corners": []}

                        for c in corners:
                            corner_data: dict[str, Any] = {"label": c["label"], "x": c["x"], "y": c["y"], "z": c["z"]}
                            if c["px"] is not None and c["py"] is not None:
                                corner_data["px"], corner_data["py"] = c["px"], c["py"]
                                if actual_active:
                                    world_points_3d.append([float(c["x"]), float(c["y"]), float(c["z"])])
                                    pixel_points.append([float(c["px"]), float(c["py"])])
                            rect_data["corners"].append(corner_data)
                        world_rects.append(rect_data)

            except Exception as e:
                logger.error(f"Fehler beim Laden der Vierecke für {camera_name}: {e}")
            finally:
                self.system_db.db_pool.putconn(conn)

            try:
                cam_settings: dict[str, Any] = self.system_db.get_camera_settings(camera_name)
                res = cam_settings.get("resolution", [1920, 1080])
                active_prof = str(cam_settings.get("active_lens_profile", "default"))
                lens_profs: dict[str, dict[str, Any]] = self.system_db.get_all_lens_profiles()

                cam_matrix = lens_profs.get(active_prof, {}).get("camera_matrix", None)
                dist_c = lens_profs.get(active_prof, {}).get("dist_coeffs", None)

                import json
                if isinstance(cam_matrix, str):
                    try:
                        cam_matrix = json.loads(cam_matrix)
                    except Exception:
                        cam_matrix = None
                if isinstance(dist_c, str):
                    try:
                        dist_c = json.loads(dist_c)
                    except Exception:
                        dist_c = None
            except Exception as e:
                logger.error(f"Fehler beim Laden der Linsenprofile für {camera_name}: {e}")

        from server.core.math.GeometryMath import GeometryMath
        import cv2

        pose: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = GeometryMath.get_camera_pose(
            [], [], custom_rectangles=world_rects,
            img_size=(res[0], res[1]), camera_matrix_override=np.array(cam_matrix) if cam_matrix else None, 
            dist_coeffs_override=np.array(dist_c) if dist_c else None,
            camera_name=camera_name
        )

        cam_3d_data: dict[str, Any] = {}
        if pose:
            rvec, tvec, K_out, dist_out = pose
            R, _ = cv2.Rodrigues(rvec)
            sy: float = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
            singular: bool = sy < 1e-6

            yaw: float = float(np.arctan2(R[1, 0], R[0, 0]) if not singular else 0)
            pitch: float = float(np.arctan2(-R[2, 0], sy))
            pos_calc: np.ndarray = -np.dot(R.T, tvec)

            Rt: np.ndarray = np.hstack((R, tvec.reshape(3, 1)))
            P_matrix: np.ndarray = np.dot(K_out, Rt)

            cam_3d_data = {
                "pos": pos_calc.flatten(), "yaw": yaw, "pitch": pitch,
                "R_inv": R.T, "K": K_out, "dist": dist_out, "P_matrix": P_matrix
            }
            logger.info(f"✅ POSE CACHED: {camera_name} | Res: {res[0]}x{res[1]} | Profil: '{active_prof}'")

        if camera_name not in self.config_cache:
            self.config_cache[camera_name] = {}
        self.config_cache[camera_name].update({
            "pixel_points": pixel_points,
            "world_points_3d": world_points_3d,
            "room_dimensions": room_dims,
            "custom_rectangles": copy.deepcopy(world_rects),
            "resolution": res,
            "camera_matrix": cam_matrix,
            "dist_coeffs": dist_c,
            "active_lens_profile": active_prof,
            "cam_3d_data": cam_3d_data
        })

    def register_client_logic(self, name: str) -> None:
        if self.connector and self.dashboard:
            ip: str = self.connector.get_client_ip(name)
            self.dashboard.sig_register_client.emit(name, ip)
            logger.info(f"Logik: {name} ({ip}) registriert.")

    def update_heartbeat_logic(self, name: str) -> None:
        if self.dashboard:
            self.dashboard.update_heartbeat(name)

    def log_to_dashboard(self, message: str) -> None:
        if self.dashboard:
            self.dashboard.log_message(message)

    def client_offline_logic(self, name: str) -> None:
        if self.dashboard:
            self.dashboard.set_client_offline(name)

    def learn_identity(self, name: str) -> None:
        if hasattr(self.tracker, 'register_identity_at_center'):
            result: str = getattr(self.tracker, 'register_identity_at_center')(name)
            self.log_to_dashboard(result)