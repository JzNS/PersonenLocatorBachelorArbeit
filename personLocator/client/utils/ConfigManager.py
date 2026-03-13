import json
import socket
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

# Konstanten für Discovery
DISCOVERY_MSG = b"DISCOVER_SERVER"
RESPONSE_MSG = b"IAM_SERVER"
DEFAULT_PORT = 65432


class ConfigManager:
    CONFIG_DIR = Path.home() / "Documents" / "4.PersonenFinderConfig"
    CONFIG_FILE = CONFIG_DIR / "client_config.json"
    SNAPSHOT_FILE = CONFIG_DIR / "calibration_snapshot.png"

    _sender = None
    _ram_cache = {}

    @staticmethod
    def set_network_sender(sender):
        """Wird beim Start aufgerufen, damit der ConfigManager funken kann."""
        ConfigManager._sender = sender

    @staticmethod
    def set_cached_config(config_data: dict):
        """Wird aufgerufen, wenn der Server als Antwort auf DB_REQUEST_CONFIG die Daten schickt."""
        ConfigManager._ram_cache = config_data
        logging.info("ConfigManager: RAM-Cache erfolgreich mit Server-Daten aktualisiert.")

    @staticmethod
    def _sanitize_for_json(obj):
        """Macht NumPy-Daten fit für den Netzwerk-Versand."""
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: ConfigManager._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [ConfigManager._sanitize_for_json(i) for i in obj]
        return str(obj) if obj is not None else None

    # ==========================================
    # Lesezugfriff aus RAM (Cache) laden
    # ==========================================

    @staticmethod
    def load_camera_config() -> Dict[str, Any]:
        return ConfigManager._ram_cache

    @staticmethod
    def load_camera_points(camera_name: str) -> list:
        """Holt den Hauptraum (MAIN_ROOM_CALIB) nun direkt und sicher aus der Server-Datenbank."""
        all_configs = ConfigManager.load_camera_config()
        global_rects = all_configs.get("Camera_ALL", {}).get("reference_rectangles", [])
        cam_pixels = all_configs.get(camera_name, {}).get("rectangle_pixels", {})

        main_room = next((r for r in global_rects if r.get("internal_id") == "MAIN_ROOM_CALIB"), None)
        if not main_room: return []

        pixel_info = cam_pixels.get("MAIN_ROOM_CALIB", [])
        result = []
        for i, c in enumerate(main_room.get("corners", [])):
            pt = dict(c)
            if i < len(pixel_info):
                pt["px"] = pixel_info[i].get("px")
                pt["py"] = pixel_info[i].get("py")
            result.append(pt)
        return result

    @staticmethod
    def save_camera_coordinates(camera_name: str, points_3d: list, resolution: list, room_dims: dict = None) -> None:
        """Baut das 8-Punkte-Dictionary und sendet es via Netzwerk-API an den Server."""
        if len(points_3d) != 8:
            logging.error("Fehler: 8 Punkte für Raumkalibrierung erwartet.")
            return

        settings_update = {
            "coordinates": {
                "rear_plane": {
                    "lower_left": points_3d[0], "lower_right": points_3d[1],
                    "upper_left": points_3d[2], "upper_right": points_3d[3]
                },
                "front_plane": {
                    "lower_left": points_3d[4], "upper_left": points_3d[5],
                    "lower_right": points_3d[6], "upper_right": points_3d[7]
                }
            },
            "resolution": resolution
        }

        if room_dims:
            settings_update["room_dimensions"] = room_dims

        ConfigManager.update_camera_settings(camera_name, settings_update)
    @staticmethod
    def get_camera_settings(camera_name: str) -> Dict[str, Any]:
        """Gibt die Kamera-Settings zurück, kombiniert aus RAM-Cache und Standardwerten."""
        settings = ConfigManager._ram_cache.get(camera_name, {})
        defaults = {
            "camera_index": 0, "zoom": 1.0, "resolution": [1920, 1080], "target_fps": 30,
            "render_capacity": 100, "performance_mode": False,
            "view_render_enabled": True, "view_show_real": True,
            "view_show_grid": True, "view_show_skeleton": True,
            "view_show_floor_grid": True, "view_show_rays": True, "view_show_sightlines": True,
            "dead_zones": [], "mirror_zones": []
        }
        final_settings = defaults.copy()
        final_settings.update(settings)
        return final_settings

    @staticmethod
    def load_global_rectangles() -> list:
        """Holt die 3D-Koordinaten der Referenz-Vierecke aus dem RAM-Cache (der vom Server aktualisiert wird)."""
        return ConfigManager._ram_cache.get("Camera_ALL", {}).get("reference_rectangles", [])

    @staticmethod
    def load_camera_rectangle_pixels(camera_name: str) -> dict:
        """Holt die 2D-Pixel-Klicks der Kamera aus dem RAM-Cache (der vom Server aktualisiert wird)."""
        return ConfigManager._ram_cache.get(camera_name, {}).get("rectangle_pixels", {})

    # ==========================================
    # Schreibe zugriff direkt via RAM-Cache und sende Updates an Server
    # ==========================================

    @staticmethod
    def update_camera_settings(camera_name: str, new_settings: dict):
        """Aktualisiert Zonen, Checkboxen, Render-Capacity etc."""
        safe_settings = ConfigManager._sanitize_for_json(new_settings)

        # 1. Lokalen RAM-Cache sofort updaten
        if camera_name not in ConfigManager._ram_cache:
            ConfigManager._ram_cache[camera_name] = {}
        ConfigManager._ram_cache[camera_name].update(safe_settings)

        if ConfigManager._sender:
            full_settings = ConfigManager.get_camera_settings(camera_name)
            ConfigManager._sender.send_db_camera_settings(camera_name, full_settings)
        else:
            logging.error("Offline: Kamera-Settings konnten nicht an Server gesendet werden!")

    @staticmethod
    def save_global_rectangles(rectangles_data: list) -> None:
        """Sendet die echten 3D-Koordinaten in die Server-DB."""
        clean_rects = []
        for r in rectangles_data:
            clean_rects.append({
                "internal_id": r["internal_id"],
                "display_id": r["display_id"],
                "type": r["type"],
                "size_cm": r["size_cm"],
                "is_active": r.get("is_active", True),
                "corners": [{"label": c["label"], "x": c.get("x", 0.0), "y": c.get("y", 0.0), "z": c.get("z", 0.0)} for
                            c in r.get("corners", [])]
            })

        # 1. RAM Cache updaten
        if "Camera_ALL" not in ConfigManager._ram_cache: ConfigManager._ram_cache["Camera_ALL"] = {}
        ConfigManager._ram_cache["Camera_ALL"]["reference_rectangles"] = clean_rects

        # 2. An Server senden
        if ConfigManager._sender:
            ConfigManager._sender.send_db_global_rectangles(clean_rects)

    @staticmethod
    def save_camera_rectangle_pixels(camera_name: str, rectangles_data: list) -> None:
        """Sendet die X/Y Pixel-Klicks der Kamera in die Server-DB."""
        pixel_dict = {}
        for r in rectangles_data:
            pixels = [{"px": c.get("px"), "py": c.get("py")} for c in r.get("corners", [])]
            pixel_dict[r["internal_id"]] = pixels

        # 1. RAM Cache updaten
        if camera_name not in ConfigManager._ram_cache: ConfigManager._ram_cache[camera_name] = {}
        ConfigManager._ram_cache[camera_name]["rectangle_pixels"] = pixel_dict

        # 2. An Server senden
        if ConfigManager._sender:
            ConfigManager._sender.send_db_camera_pixels(camera_name, pixel_dict)

    # ==========================================
    # Network Config (Local auf Client)
    # ==========================================
    @staticmethod
    def load_config() -> Tuple[str, str, int]:
        """Lädt die Client-Konfiguration (Name, Server-IP, Port)
        aus der lokalen JSON-Datei. Wenn die Datei fehlt oder fehlerhaft ist,
        werden Standardwerte zurückgegeben."""
        if not ConfigManager.CONFIG_FILE.exists():
            return "Client_Neu", "AUTO", DEFAULT_PORT
        try:
            with ConfigManager.CONFIG_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get("client_name", "Unknown_Client"), data.get("server_ip", "AUTO"), data.get("server_port",
                                                                                                      DEFAULT_PORT)
        except Exception:
            return "Error_Client", "AUTO", DEFAULT_PORT

    @staticmethod
    def find_server_ip(port: int = DEFAULT_PORT, timeout: float = 5.0) -> Optional[str]:
        """Versucht, die Server-IP durch einen UDP-Broadcast zu ermitteln.
        Sendet eine DISCOVERY-Nachricht und wartet auf die Antwort."""
        logging.info("Suche Server via UDP Broadcast...")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_sock.settimeout(timeout)
            try:
                udp_sock.sendto(DISCOVERY_MSG, ('<broadcast>', port))
                data, addr = udp_sock.recvfrom(1024)
                if data == RESPONSE_MSG:
                    return addr[0]
            except Exception:
                pass
        return None