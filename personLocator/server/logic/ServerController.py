import logging
import numpy as np
from pathlib import Path
from typing import Optional
from psycopg2.extras import RealDictCursor
from server.gui.logic.GlobalPerson import GlobalPerson
from server.gui.logic.ServerPersonTracker import ServerPersonTracker
from server.gui.logic.data.PersonDatabase import PersonDatabase
from server.network.logic.ServerCommandHandler import ServerCommandHandler
from server.network.ServerConnector import ServerConnector
from server.network.logic.ServerBeacon import ServerBeacon
from server.network.logic.ServerCommandSender import ServerCommandSender
from server.gui.logic.data.SystemDatabase import SystemDatabase


class ServerController:
    """
    Zentrale Steuereinheit (Mediator).
    Verbindet Netzwerk, Tracker und Dashboard.
    """

    def __init__(self, port: int, dashboard=None) -> None:
        self.port = port
        self.dashboard = dashboard
        if self.dashboard:
            self.dashboard.set_controller(self)

        self.person_db = PersonDatabase()
        self.system_db = SystemDatabase()
        self.config_cache = {}

        self.tracker = ServerPersonTracker(
            person_database=self.person_db,
            person_class_ref=GlobalPerson
        )

        self.connector = ServerConnector(port=self.port)
        self.beacon = ServerBeacon(port=self.port)
        self.sender = ServerCommandSender(self.connector)

        self.handler = ServerCommandHandler(
            server_connector=self.connector,
            command_sender=self.sender,
            controller=self
        )
        self.connector.set_command_handler(self.handler)
        self.refresh_config_cache()

    def start(self) -> None:
        self.connector.start()
        self.beacon.start()
        if self.dashboard:
            self.dashboard.log_message(f"Server gestartet auf Port {self.port}")

    def refresh_config_cache(self):
        """Lädt den Cache neu und triggert GUI-Updates."""
        try:
            self.config_cache.clear()

            self.log_to_dashboard("🔄 System-Cache wurde nach DB-Update aktualisiert.")
            if self.dashboard:
                self.dashboard.sig_update_camera_view.emit("SYSTEM_REFRESH", [])

            if hasattr(self, 'db_viewer') and self.db_viewer and self.db_viewer.isVisible():
                self.db_viewer.external_data_updated()

        except Exception as e:
            logging.error(f"Fehler beim Cache-Refresh: {e}")

    def execute_client_handshake(self, client_name: str) -> None:
        """Handshakes sind die magischen Momente, in denen ein Client zum ersten Mal mit dem Server spricht.
        Hier entscheiden wir, ob er die volle 3D-Konfiguration oder die Lite-Version bekommt."""
        if not hasattr(self, 'system_db'): return
        try:
            self.system_db.register_camera_if_not_exists(client_name)
            settings = self.system_db.get_camera_settings(client_name)

            if not settings:
                logging.warning(f"Konnte Settings für {client_name} nicht laden. Sende Lite-Config.")
                self.sender.send_message(client_name, "CONFIG_LITE", {})
                return

            perf_mode = settings.get("performance_mode", False)

            if perf_mode:
                self.log_to_dashboard(f"Handshake: {client_name} ist im Performance-Modus.")
                payload = {
                    client_name: settings,
                    "Camera_ALL": self.system_db.get_all_lens_profiles()
                }
                self.sender.send_message(client_name, "CONFIG_LITE", payload)
            else:
                self.log_to_dashboard(f"Handshake: {client_name} ist im 3D-Modus.")
                self.send_full_config_to_client(client_name)

        except Exception as e:
            self.log_to_dashboard(f"Fehler beim Handshake mit {client_name}: {e}")

    def sync_all_clients(self) -> None:
        """Sobald sich die DB ändert, müssen alle Clients ihre Konfiguration aktualisieren.
        Das ist der Moment, in dem wir die Magie des Live-Syncs entfesseln!"""
        if not hasattr(self, 'connector') or not self.connector.clients:
            return

        self.log_to_dashboard(f"🔄 Live-Sync: Sende DB-Update an {len(self.connector.clients)} verbundene Clients...")
        for client_name in list(self.connector.clients.keys()):
            self.execute_client_handshake(client_name)

    def send_full_config_to_client(self, client_name: str) -> None:
        """Sendet die vollständige 3D-Konfiguration an den Client. Das ist der Moment, in dem wir die volle Power der DB-Integration zeigen!"""
        try:
            settings = self.system_db.get_camera_settings(client_name)
            global_rects = self.system_db.get_all_world_rectangles()
            pixels = self.system_db.get_camera_pixels(client_name)

            payload = {
                client_name: settings,
                "Camera_ALL": {
                    "reference_rectangles": global_rects,
                    "room_dimensions": self.system_db.get_global_room()
                }
            }
            payload[client_name]["rectangle_pixels"] = pixels
            payload["Camera_ALL"].update(self.system_db.get_all_lens_profiles())

            self.sender.send_message(client_name, "CONFIG_FULL", payload)
        except Exception as e:
            self.log_to_dashboard(f"Fehler CONFIG_FULL für {client_name}: {e}")

    def get_unmerged_tracks(self):
        return self.tracker.get_raw_tracks()

    def execute_manual_merge(self, name, selection_list):
        msg = self.tracker.force_merge_and_calibrate(name, selection_list)
        self.log_to_dashboard(msg)

    def update_camera_view_logic(self, camera_name: str, person_list: list):
        from server.gui.logic.GeometryMath import GeometryMath

        # ==========================================
        # 1. DB Abfrage nur für Cache, ganz am start einmal
        # ==========================================
        if camera_name not in self.config_cache:
            self._fetch_camera_data_from_db(camera_name)

        cam_conf = self.config_cache.get(camera_name, {})
        pixel_points = cam_conf.get("pixel_points", [])
        world_points_3d = cam_conf.get("world_points_3d", [])
        world_rects = cam_conf.get("custom_rectangles", [])
        res = cam_conf.get("resolution", [1920, 1080])
        cam_matrix = cam_conf.get("camera_matrix")
        dist_c = cam_conf.get("dist_coeffs")

        # ==========================================
        # 2. Personen 3D-Ortung durchführen
        # ==========================================
        clean_persons = []
        for p in person_list:
            bbox = p.get('bbox', [])
            kps = p.get('keypoints', [])
            calc_pos = None

            if len(pixel_points) >= 4 and len(world_points_3d) >= 4 and len(bbox) == 4:
                calc_pos = GeometryMath.smart_project_position(
                    kps, bbox, pixel_points, world_points_3d, 180.0, world_rects,
                    img_size=res, camera_matrix_override=cam_matrix, dist_coeffs_override=dist_c
                )

            p['pos'] = calc_pos if calc_pos is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
            clean_persons.append(p)

        self.tracker.update_camera_data(camera_name, clean_persons)

        # ==========================================
        # 3. GUI Updates
        # ==========================================
        if self.dashboard:
            self.dashboard.update_camera_data(camera_name, clean_persons)

            export_list = [{"id": gp.id, "pos": gp.pos, "height": gp.height, "status": gp.name, "keypoints": []}
                           for gp in self.tracker.global_persons]
            self.dashboard.update_camera_data("MASTER_FUSION", export_list)

    def _fetch_camera_data_from_db(self, camera_name: str):
        """Holt die Daten EXAKT EINMAL aus der DB und legt sie im pfeilschnellen RAM-Cache ab."""
        pixel_points, world_points_3d, world_rects = [], [], []
        room_dims = {"width": 600, "height": 250, "depth": 800}
        res, cam_matrix, dist_c = [1920, 1080], None, None

        if hasattr(self, 'system_db') and self.system_db.db_pool:
            conn = self.system_db.db_pool.getconn()
            try:

                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # 1. Raummaße
                    cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                    r = cursor.fetchone()
                    if r: room_dims = {"width": r["width_cm"], "height": r["height_cm"], "depth": r["depth_cm"]}

                    # 2. Vierecke
                    cursor.execute("SELECT * FROM world_rectangles")
                    for rect in cursor.fetchall():
                        r_id = rect["id"]
                        cursor.execute("""
                            SELECT c.label, c.x, c.y, c.z, m.px, m.py
                            FROM rectangle_corners_3d c
                            LEFT JOIN camera_pixel_mapping m ON m.corner_3d_id = c.id AND m.camera_name = %s
                            WHERE c.rect_id = %s ORDER BY c.id ASC
                        """, (camera_name, r_id))

                        corners = cursor.fetchall()
                        rect_data = {"internal_id": r_id, "display_id": rect["display_id"], "type": rect["type"],
                                     "is_active": rect["is_active"], "corners": []}

                        for c in corners:
                            corner_data = {"label": c["label"], "x": c["x"], "y": c["y"], "z": c["z"]}
                            if c["px"] is not None and c["py"] is not None:
                                corner_data["px"], corner_data["py"] = c["px"], c["py"]
                                if rect["is_active"]:
                                    world_points_3d.append([c["x"], c["y"], c["z"]])
                                    pixel_points.append([c["px"], c["py"]])
                            rect_data["corners"].append(corner_data)
                        world_rects.append(rect_data)

                # 3. Linsenprofil
                cam_settings = self.system_db.get_camera_settings(camera_name)
                res = cam_settings.get("resolution", [1920, 1080])
                active_prof = cam_settings.get("active_lens_profile", "default")
                lens_profs = self.system_db.get_all_lens_profiles()
                cam_matrix = lens_profs.get(active_prof, {}).get("camera_matrix", None)
                dist_c = lens_profs.get(active_prof, {}).get("dist_coeffs", None)

            except Exception as e:
                logging.error(f"Fehler beim Laden der DB-Daten für {camera_name}: {e}")
            finally:
                self.system_db.db_pool.putconn(conn)

        # In den RAM schreiben
        if camera_name not in self.config_cache: self.config_cache[camera_name] = {}
        self.config_cache[camera_name].update({
            "pixel_points": pixel_points, "world_points_3d": world_points_3d,
            "room_dimensions": room_dims, "custom_rectangles": world_rects,
            "resolution": res, "camera_matrix": cam_matrix, "dist_coeffs": dist_c
        })
    def register_client_logic(self, name: str) -> None:
        if self.connector and self.dashboard:
            ip = self.connector.get_client_ip(name)
            self.dashboard.register_client(name, ip)
            logging.info(f"Logik: {name} ({ip}) registriert.")

    def update_heartbeat_logic(self, name: str) -> None:
        if self.dashboard:
            self.dashboard.update_heartbeat(name)

    def log_to_dashboard(self, message: str) -> None:
        if self.dashboard:
            self.dashboard.log_message(message)

    def client_offline_logic(self, name: str) -> None:
        if self.dashboard:
            self.dashboard.set_client_offline(name)

    def learn_identity(self, name):
        result = self.tracker.register_identity_at_center(name)
        self.log_to_dashboard(result)