import json
import logging
from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
    MASTER_CONFIG_PATH = Path(r"server/config/camera_config.json")
    POINT_ORDER = [
        ("rear_plane", "lower_left"),  # 0: HUL
        ("rear_plane", "lower_right"),  # 1: HUR
        ("rear_plane", "upper_left"),  # 2: HOL
        ("rear_plane", "upper_right"),  # 3: HOR
        ("front_plane", "lower_left"),  # 4: VUL
        ("front_plane", "upper_left"),  # 5: VOL
        ("front_plane", "lower_right"),  # 6: VUR
        ("front_plane", "upper_right")  # 7: VOR
    ]

    @staticmethod
    def get_master_config_date() -> str:
        """Liest nur das Datum, ohne die ganze Datei zu parsen."""
        if not ConfigManager.MASTER_CONFIG_PATH.exists():
            return "1970-01-01 00:00:00"
        try:
            with ConfigManager.MASTER_CONFIG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("last_update", "1970-01-01 00:00:00")
        except Exception as e:
            logging.error(f"Fehler beim Datum-Lesen: {e}")
            return "1970-01-01 00:00:00"

    @staticmethod
    def get_master_config_data() -> Dict[str, Any]:
        """
        Liest die Config und ERSTELLT 'world_points_3d' UND 'pixel_points'
        dynamisch aus 'coordinates'.
        """
        if not ConfigManager.MASTER_CONFIG_PATH.exists():
            return {}

        try:
            with ConfigManager.MASTER_CONFIG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)

            for cam_key, cam_data in data.items():
                if not cam_key.startswith("CAMERA_"): continue

                if "coordinates" in cam_data:
                    w_list = ConfigManager._generate_world_points(cam_data["coordinates"])
                    if w_list:
                        cam_data["world_points_3d"] = w_list
                    p_list = ConfigManager._generate_pixel_points(cam_data["coordinates"])
                    if p_list:
                        cam_data["pixel_points"] = p_list

            return data

        except Exception as e:
            logging.error(f"Fehler beim Laden der Master-Daten: {e}")
            return {}


    @staticmethod
    def _generate_world_points(coords: Dict) -> List[List[float]]:
        """Zieht x, y, z aus der Struktur und wandelt sie in eine sortierte Liste um."""
        points_list = []
        try:
            for plane, corner in ConfigManager.POINT_ORDER:
                pt = coords[plane][corner]
                points_list.append([float(pt["x"]), float(pt["y"]), float(pt["z"])])
            return points_list
        except (KeyError, ValueError):
            return []

    @staticmethod
    def _generate_pixel_points(coords: Dict) -> List[List[int]]:
        """Zieht px und py aus der Struktur."""
        pixels_list = []
        try:
            for plane, corner in ConfigManager.POINT_ORDER:
                pt = coords[plane][corner]
                px = int(pt.get("px", 0))
                py = int(pt.get("py", 0))
                pixels_list.append([px, py])
            return pixels_list
        except (KeyError, ValueError):
            return []

    @staticmethod
    def update_master_config_data(new_data: Dict[str, Any]) -> None:
        """
        Speichert die Config, aber LÖSCHT vorher 'world_points_3d',
        damit die Datei klein und fehlerfrei bleibt.
        """
        try:
            import copy
            data_to_save = copy.deepcopy(new_data)

            for cam_key, cam_data in data_to_save.items():
                if isinstance(cam_data, dict) and "world_points_3d" in cam_data:
                    del cam_data["world_points_3d"]

            if not ConfigManager.MASTER_CONFIG_PATH.parent.exists():
                ConfigManager.MASTER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

            with ConfigManager.MASTER_CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)

            logging.info("SERVER: Master-Config gespeichert (world_points_3d bereinigt).")

        except Exception as e:
            logging.error(f"SERVER: Fehler beim Speichern: {e}")

    @staticmethod
    def _generate_points_from_coords(coords: Dict) -> List[List[float]]:
        """Wandelt das Coordinates-Dict in die sortierte Liste um."""
        points_list = []
        try:
            for plane, corner in ConfigManager.POINT_ORDER:
                pt = coords[plane][corner]
                points_list.append([
                    float(pt["x"]),
                    float(pt["y"]),
                    float(pt["z"])
                ])
            return points_list
        except KeyError as e:
            logging.warning(f"Config unvollständig, konnte Punkt nicht finden: {e}")
            return []
        except ValueError:
            logging.warning("Config enthält ungültige Zahlenwerte.")
            return []