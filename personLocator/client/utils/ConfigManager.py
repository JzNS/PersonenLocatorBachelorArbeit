import json
import socket
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np

DISCOVERY_MSG: bytes = b"DISCOVER_SERVER"
RESPONSE_MSG: bytes = b"IAM_SERVER"
DEFAULT_PORT: int = 45432


class ConfigManager:
    CONFIG_DIR: Path = Path.home() / "Documents" / "4.PersonenFinderConfig"
    CONFIG_FILE: Path = CONFIG_DIR / "client_config.json"

    _sender: Optional[Any] = None
    _ram_cache: dict[str, Any] = {}
    _is_locked: bool = False

    @staticmethod
    def get_snapshot_file(camera_name: str) -> Path:
        """Generiert einen spezifischen Dateipfad für jede Kamera."""
        return ConfigManager.CONFIG_DIR / f"calibration_snapshot_{camera_name}.png"

    @staticmethod
    def set_network_sender(sender: Any) -> None:
        """Wird beim Start aufgerufen, damit der ConfigManager funken kann."""
        ConfigManager._sender = sender

    @staticmethod
    def set_edit_mode(active: bool) -> None:
        """Sperrt oder entsperrt den Config-Cache für Server-Updates."""
        ConfigManager._is_locked = active
        logging.info(f"ConfigManager: Edit-Modus {'AKTIVIERT' if active else 'DEAKTIVIERT'}")

    @staticmethod
    def save_lens_profile(camera_name: str, profile_id: str, name: str, mtx: list[Any], dist: list[Any], reprojection_error: float = None) -> None:
        """Speichert ein neues Linsenprofil global und wendet es auf die aktuelle Kamera an."""
        print("ConfigManager: Speichere Linsenprofil - Kamera:", camera_name, "Profil-ID:", profile_id)
        payload: dict[str, Any] = {
            "profile_id": profile_id,
            "name": name,
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "reprojection_error": reprojection_error
        }

        if ConfigManager._sender and hasattr(ConfigManager._sender, 'send_db_lens_profile'):
            getattr(ConfigManager._sender, 'send_db_lens_profile')(payload)

            import time
            time.sleep(0.5)  # Kurze Pause, damit die DB das Profil anlegen kann
        else:
            logging.error("Offline: Linsenprofil konnte nicht an Server gesendet werden!")

        ConfigManager.update_camera_settings(camera_name, {"active_lens_profile": profile_id})

    @staticmethod
    def set_cached_config(config_data: dict[str, Any]) -> None:
        if ConfigManager._is_locked:
            logging.debug("ConfigManager: Sync ignoriert, da Edit-Modus aktiv.")
            return

        ConfigManager._ram_cache = config_data
        logging.info("ConfigManager: RAM-Cache aktualisiert.")

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Macht NumPy-Daten fit für den Netzwerk-Versand."""
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): ConfigManager._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ConfigManager._sanitize_for_json(i) for i in obj]
        return str(obj) if obj is not None else None

    @staticmethod
    def load_camera_config() -> dict[str, Any]:
        return ConfigManager._ram_cache

    @staticmethod
    def load_camera_points(camera_name: str) -> list[dict[str, Any]]:
        """Holt den Hauptraum (MAIN_ROOM_CALIB) nun direkt und sicher aus der Server-Datenbank."""
        all_configs: dict[str, Any] = ConfigManager.load_camera_config()
        global_rects: list[dict[str, Any]] = all_configs.get("Camera_ALL", {}).get("reference_rectangles", [])
        cam_pixels: dict[str, Any] = all_configs.get(camera_name, {}).get("rectangle_pixels", {})

        main_room: Optional[dict[str, Any]] = next((r for r in global_rects if r.get("internal_id") == "MAIN_ROOM_CALIB"), None)
        if not main_room:
            return []

        cam_data: Any = cam_pixels.get("MAIN_ROOM_CALIB", {})
        pixel_info: list[Any]
        if isinstance(cam_data, dict) and "pixels" in cam_data:
            pixel_info = cam_data.get("pixels", [])
        else:
            pixel_info = cam_data if isinstance(cam_data, list) else []

        result: list[dict[str, Any]] = []
        for i, c in enumerate(main_room.get("corners", [])):
            pt: dict[str, Any] = dict(c)
            if i < len(pixel_info):
                pt["px"] = pixel_info[i].get("px")
                pt["py"] = pixel_info[i].get("py")
            result.append(pt)
        return result

    @staticmethod
    def save_camera_coordinates(camera_name: str, points_3d: list[Any], resolution: list[int], room_dims: Optional[dict[str, float]] = None) -> None:
        """Baut das 8-Punkte-Dictionary und sendet es via Netzwerk-API an den Server."""
        if len(points_3d) != 8:
            logging.error("Fehler: 8 Punkte für Raumkalibrierung erwartet.")
            return

        settings_update: dict[str, Any] = {
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
    def get_camera_settings(camera_name: Union[str, list[str]]) -> dict[str, Any]:
        """Gibt die Kamera-Settings zurück, kombiniert aus RAM-Cache und Standardwerten."""

        clean_name: str
        if isinstance(camera_name, list):
            clean_name = str(camera_name[0]) if camera_name else "Unknown_Camera"
            logging.warning(f"Typkorrektur: Liste empfangen, nutze '{clean_name}' als Key.")
        else:
            clean_name = str(camera_name).strip()

        settings: dict[str, Any] = ConfigManager._ram_cache.get(clean_name, {})

        defaults: dict[str, Any] = {
            "camera_index": 0, "zoom": 1.0, "resolution": [1920, 1080], "target_fps": 30,
            "render_capacity": 100, "performance_mode": False,
            "view_render_enabled": True, "view_show_real": True,
            "view_show_grid": True, "view_show_skeleton": True,
            "view_show_floor_grid": True, "view_show_rays": True, "view_show_sightlines": True,
            "dead_zones": [], "mirror_zones": []
        }

        final_settings: dict[str, Any] = defaults.copy()
        final_settings.update(settings)
        return final_settings

    @staticmethod
    def load_global_rectangles() -> list[dict[str, Any]]:
        """Holt die 3D-Koordinaten der Referenz-Vierecke aus dem RAM-Cache (der vom Server aktualisiert wird)."""
        return list(ConfigManager._ram_cache.get("Camera_ALL", {}).get("reference_rectangles", []))

    @staticmethod
    def load_camera_rectangle_pixels(camera_name: str) -> dict[str, Any]:
        """Holt die 2D-Pixel-Klicks der Kamera aus dem RAM-Cache (der vom Server aktualisiert wird)."""
        return dict(ConfigManager._ram_cache.get(camera_name, {}).get("rectangle_pixels", {}))

    @staticmethod
    def update_camera_settings(camera_name: str, new_settings: dict[str, Any]) -> None:
        """Aktualisiert Zonen, Checkboxen, Render-Capacity etc."""
        safe_settings: Any = ConfigManager._sanitize_for_json(new_settings)

        if camera_name not in ConfigManager._ram_cache:
            ConfigManager._ram_cache[camera_name] = {}
        ConfigManager._ram_cache[camera_name].update(safe_settings)

        if ConfigManager._sender:
            full_settings: dict[str, Any] = ConfigManager.get_camera_settings(camera_name)
            getattr(ConfigManager._sender, 'send_db_camera_settings')(camera_name, full_settings)
        else:
            logging.error("Offline: Kamera-Settings konnten nicht an Server gesendet werden!")

    @staticmethod
    def save_global_rectangles(rectangles_data: list[dict[str, Any]]) -> None:
        """Sendet die echten 3D-Koordinaten in die Server-DB."""
        clean_rects: list[dict[str, Any]] = []
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

        if "Camera_ALL" not in ConfigManager._ram_cache:
            ConfigManager._ram_cache["Camera_ALL"] = {}
        ConfigManager._ram_cache["Camera_ALL"]["reference_rectangles"] = clean_rects

        if ConfigManager._sender:
            getattr(ConfigManager._sender, 'send_db_global_rectangles')(clean_rects)

    @staticmethod
    def save_camera_rectangle_pixels(camera_name: str, rectangles_data: list[dict[str, Any]]) -> None:
        """Sendet die X/Y Pixel-Klicks der Kamera in die Server-DB."""
        pixel_dict: dict[str, Any] = {}
        for r in rectangles_data:
            pixels: list[dict[str, Any]] = [{"px": c.get("px"), "py": c.get("py")} for c in r.get("corners", [])]

            pixel_dict[r["internal_id"]] = {
                "is_active": r.get("is_active", True),
                "pixels": pixels
            }

        if camera_name not in ConfigManager._ram_cache:
            ConfigManager._ram_cache[camera_name] = {}
        ConfigManager._ram_cache[camera_name]["rectangle_pixels"] = pixel_dict

        if ConfigManager._sender:
            getattr(ConfigManager._sender, 'send_db_camera_pixels')(camera_name, pixel_dict)

    @staticmethod
    def load_config() -> Tuple[list[str], str, int]:
        """Lädt die Client-Konfiguration (Name, Server-IP, Port)
        aus der lokalen JSON-Datei. Wenn die Datei fehlt oder fehlerhaft ist,
        werden Standardwerte zurückgegeben.
        Gibt 'client_name' als Liste zurück, falls es im JSON eine Liste ist."""
        if not ConfigManager.CONFIG_FILE.exists():
            return ["Client_Neu"], "AUTO", DEFAULT_PORT
        try:
            with ConfigManager.CONFIG_FILE.open("r", encoding="utf-8") as file:
                data: dict[str, Any] = json.load(file)

            client_name_data: Any = data.get("client_name", ["Unknown_Client"])

            if isinstance(client_name_data, str):
                client_name_data = [client_name_data]

            server_port: int = int(data.get("server_port", DEFAULT_PORT))

            if server_port == 65432:
                logging.warning(f"Port 65432 ist als blockiert bekannt. Migriere automatisch auf {DEFAULT_PORT}.")
                server_port = DEFAULT_PORT

            return list(client_name_data), str(data.get("server_ip", "AUTO")), server_port
        except Exception:
            return ["Error_Client"], "AUTO", DEFAULT_PORT

    @staticmethod
    def find_server_ip(port: int = DEFAULT_PORT, timeout: float = 5.0) -> Optional[str]:
        """Versucht, die Server-IP durch einen UDP-Broadcast zu ermitteln.
        Sendet eine DISCOVERY-Nachricht und wartet auf die Antwort."""
        logging.info(f"Suche Server via UDP Broadcast auf Port {port}...")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_sock.settimeout(timeout)
            try:
                udp_sock.sendto(DISCOVERY_MSG, ('<broadcast>', port))
                data, addr = udp_sock.recvfrom(1024)
                if data == RESPONSE_MSG:
                    return str(addr[0])
            except Exception:
                pass
        return None