import time
import json
import os
from server.core.logger import get_logger
from typing import Any, Dict, Optional, TYPE_CHECKING

logger = get_logger("server.network.handler")

if TYPE_CHECKING:
    from server.network.ServerConnector import ServerConnector
    from server.network.ServerCommandSender import ServerCommandSender
    from server.controllers.ServerController import ServerController


class ServerCommandHandler:
    def __init__(self, server_connector: 'ServerConnector', command_sender: 'ServerCommandSender',
                 controller: 'ServerController') -> None:
        self.server_connector: 'ServerConnector' = server_connector
        self.command_sender: 'ServerCommandSender' = command_sender
        self.controller: 'ServerController' = controller

    def handle_message(self, msg: Dict[str, Any], sender: str) -> None:
        target: Optional[str] = msg.get("target")
        action: Optional[str] = msg.get("action")
        payload: Any = msg.get("payload")

        if not action:
            return

        if action.startswith("DB_"):
            logger.info(f"📩 API-Request von {sender}: {action}")

        if action == "DB_UPDATE_CAMERA_SETTINGS":
            try:
                self.controller.system_db.update_camera_settings(sender, payload)
                self.controller.refresh_config_cache()
                self.controller.log_to_dashboard(f"✅ Settings für {sender} gespeichert.")

                # Sync, damit Perf-Mode-Wechsel auf allen Clients ankommen
                if hasattr(self.controller, 'sync_all_clients'):
                    self.controller.sync_all_clients()
            except Exception as e:
                logger.error(f"Fehler bei DB_UPDATE_CAMERA_SETTINGS: {e}")

        elif action == "DB_SAVE_LENS_PROFILE":
            if isinstance(payload, dict):
                self.controller.system_db.save_lens_profile(
                    str(payload.get("profile_id", "")),
                    str(payload.get("name", "")),
                    payload.get("camera_matrix", []),
                    payload.get("dist_coeffs", [])
                )
                self.controller.refresh_config_cache()
                if hasattr(self.controller, 'sync_all_clients'):
                    self.controller.sync_all_clients()
                self._export_calibration_json()

        elif action == "DB_UPDATE_GLOBAL_RECTANGLES":
            if isinstance(payload, dict):
                self.controller.system_db.update_global_rectangles(payload.get("rectangles", []))
                self.controller.refresh_config_cache()
                if hasattr(self.controller, 'sync_all_clients'):
                    self.controller.sync_all_clients()

        elif action == "DB_UPDATE_GLOBAL_ROOM":
            if isinstance(payload, dict):
                self.controller.system_db.update_global_room(payload.get("room_dimensions", {}))
                self.controller.refresh_config_cache()
                if hasattr(self.controller, 'sync_all_clients'):
                    self.controller.sync_all_clients()

        elif action == "DB_UPDATE_CAMERA_PIXELS":
            if isinstance(payload, dict):
                cam_target: str = str(payload.get("camera", sender))
                self.controller.system_db.update_camera_pixels(cam_target, payload.get("pixels", {}))
                self.controller.refresh_config_cache()

        elif action == "DB_REQUEST_CONFIG":
            self.controller.send_full_config_to_client(sender)
            if hasattr(self.controller, 'sync_all_clients'):
                self.controller.sync_all_clients()

        elif action == "REGISTER":
            name: str = sender
            if isinstance(payload, dict):
                name = str(payload.get("name", sender))
            self.controller.register_client_logic(name)
            self.controller.execute_client_handshake(name)
            self.command_sender.send_pong(name)

        elif action == "CAMERA_UPDATE":
            if isinstance(payload, dict):
                cam_name: str = str(payload.get("camera", sender))
                t3 = time.time()  # t3 = Server-Ankunft, serverseitig gesetzt
                t2 = msg.get("t2", 0.0)
                t1 = payload.get("t1", 0.0)
                inf_time = payload.get("inference_time_ms", 0.0)
                client_fps = float(payload.get("fps", 0.0))
                self.controller.update_camera_view_logic(cam_name, payload.get("persons", []), t3=t3, t2=t2, t1=t1, inf_time=inf_time, client_fps=client_fps)

        elif action == "PING":
            self.controller.update_heartbeat_logic(sender)
            self.command_sender.send_pong(sender)

        if target == "ALL":
            self.controller.log_to_dashboard(f"Broadcast von {sender}: {payload}")
            self.command_sender.broadcast_message(sender, str(payload))

    def __handle_server_command(self, action: str, sender: str, payload: Any) -> None:
        if action == "LOG":
            self.controller.log_to_dashboard(f"LOG von {sender}: {payload}")

    def __handle_broadcast_command(self, action: str, sender: str, payload: Any) -> None:
        self.controller.log_to_dashboard(f"Broadcast {sender}: {payload}")
        self.command_sender.broadcast_message(sender, str(payload))


    def _export_calibration_json(self) -> None:
        """Schreibt alle Linsenprofile nach einer Kalibrierung in calibration.json."""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            profiles = self.controller.system_db.get_all_lens_profiles()
            export = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "lens_profiles": {}
            }
            for name, data in profiles.items():
                export["lens_profiles"][name] = {
                    "camera_matrix": data.get("camera_matrix", []),
                    "dist_coeffs": data.get("dist_coeffs", []),
                    "reprojection_error": data.get("reprojection_error", None)
                }
            cal_path = os.path.join(log_dir, "calibration.json")
            with open(cal_path, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, default=str)
            self.controller.log_to_dashboard(f"📁 calibration.json gespeichert: {cal_path}")
        except Exception as e:
            logger.error(f"Fehler beim Schreiben von calibration.json: {e}")

    def handle_client_disconnect(self, client_name: str) -> None:
        self.controller.client_offline_logic(client_name)