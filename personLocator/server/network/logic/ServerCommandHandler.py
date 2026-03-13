# server/network/logic/ServerCommandHandler.py
import logging
from typing import Any, Dict

from server.utils.ConfigManager import ConfigManager


class ServerCommandHandler:
    def __init__(self, server_connector, command_sender, controller):
        self.server_connector = server_connector
        self.command_sender = command_sender
        self.controller = controller

    def handle_message(self, msg: Dict[str, Any], sender: str):
        target = msg.get("target")
        action = msg.get("action")
        payload = msg.get("payload")
        print(f"DEBUG: Received message from {sender} -> Target: {target}, Action: {action}, Payload: {payload}")

        if action.startswith("DB_"):
            log_text = f"📩 API-Request von {sender}: {action}"
            if action == "DB_UPDATE_CAMERA_SETTINGS":
                cap = payload.get("render_capacity", "N/A")
                log_text = f"📩 API-Request von {sender}: {action} -> KI-Last: {cap}%"
                self.controller.log_to_dashboard(log_text)

                self.controller.system_db.update_camera_settings(sender, payload)
            elif action == "DB_UPDATE_GLOBAL_RECTANGLES":
                rects = payload.get("rectangles", [])
                log_text += f" -> {len(rects)} Vierecke"
            self.controller.log_to_dashboard(log_text)


        if action == "CAMERA_UPDATE":
            cam_name = payload.get("camera", sender)
            persons = payload.get("persons", [])
            self.controller.update_camera_view_logic(cam_name, persons)

        elif action == "REGISTER":
            name = payload.get("name", sender) if isinstance(payload, dict) else sender
            self.controller.register_client_logic(name)
            self.controller.execute_client_handshake(name)
            self.command_sender.send_pong(name)
        elif action == "DB_REQUEST_CONFIG":
            self.controller.send_full_config_to_client(sender)
        elif action == "DB_UPDATE_CAMERA_SETTINGS":
            try:
                self.controller.system_db.update_camera_settings(sender, payload)
                self.controller.refresh_config_cache()
            except Exception as e:
                logging.error(f"Fehler bei DB_UPDATE_CAMERA_SETTINGS: {e}")
        elif action == "REQUEST_CONFIG_FULL":
            logging.info(f"Client {sender} fordert volle 3D-Konfiguration an.")
            if hasattr(self.controller, "send_full_config_to_client"):
                self.controller.send_full_config_to_client(sender)

        elif action == "DB_UPDATE_GLOBAL_RECTANGLES":
            try:
                rects = payload.get("rectangles", [])
                self.controller.system_db.update_global_rectangles(rects)
                self.controller.refresh_config_cache()
                self.controller.log_to_dashboard("🗺️ DB-Update: 3D-Welt erfolgreich synchronisiert.")
                if hasattr(self.controller, 'sync_all_clients'):
                    self.controller.sync_all_clients()
            except Exception as e:
                logging.error(f"Schwerer Fehler bei DB_UPDATE_GLOBAL_RECTANGLES: {e}")
        elif action == "DB_UPDATE_GLOBAL_ROOM":
                try:
                    dims = payload.get("room_dimensions", {})
                    self.controller.system_db.update_global_room(dims)
                    self.controller.refresh_config_cache()
                    self.controller.log_to_dashboard("📐 DB-Update: Raummaße aktualisiert.")

                    if hasattr(self.controller, 'sync_all_clients'):
                        self.controller.sync_all_clients()
                except Exception as e:
                    logging.error(f"Fehler bei DB_UPDATE_GLOBAL_ROOM: {e}")
        elif action == "DB_UPDATE_CAMERA_PIXELS":
            try:
                cam_target = payload.get("camera", sender)
                pixels = payload.get("pixels", {})
                self.controller.system_db.update_camera_pixels(cam_target, pixels)
                self.controller.refresh_config_cache()
                self.controller.log_to_dashboard(f"🎯 DB-Update: Pixel-Mapping für {cam_target} gespeichert.")
            except Exception as e:
                logging.error(f"Schwerer Fehler bei DB_UPDATE_CAMERA_PIXELS: {e}")
        if action == "CAMERA_UPDATE":
            cam_name = payload.get("camera", sender)
            persons = payload.get("persons", [])
            self.controller.update_camera_view_logic(cam_name, persons)


        if action == "QUERY_CONFIG_DATE":
            server_date = ConfigManager.get_master_config_date()
            self.command_sender.send_message(sender, "CONFIG_DATE", server_date)

        elif action == "REQUEST_CONFIG_FULL":
            full_config = ConfigManager.get_master_config_data()
            self.command_sender.send_message(sender, "CONFIG_UPDATE", full_config)

        if action == "PING":
            self.controller.update_heartbeat_logic(sender)
            self.command_sender.send_pong(sender)

        if target == "SERVER":
            self.__handle_server_command(action, sender, payload)

        elif target == "ALL":
            self.__handle_broadcast_command(action, sender, payload)

    def __handle_server_command(self, action: str, sender: str, payload: Any):
        if action == "LOG":
            self.controller.log_to_dashboard(f"LOG von {sender}: {payload}")

    def __handle_broadcast_command(self, action: str, sender: str, payload: Any):
        self.controller.log_to_dashboard(f"Broadcast {sender}: {payload}")
        self.command_sender.broadcast_message(sender, str(payload))

    def handle_client_disconnect(self, client_name: str) -> None:
        self.controller.client_offline_logic(client_name)