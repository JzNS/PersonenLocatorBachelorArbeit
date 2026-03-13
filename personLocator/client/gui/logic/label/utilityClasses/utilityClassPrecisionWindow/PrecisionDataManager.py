import uuid
from client.utils.ConfigManager import ConfigManager


class PrecisionDataManager:
    @staticmethod
    def load_all_rectangles(camera_name: str) -> list:
        """Lädt alle Rechtecke (Hauptraum + Custom)
        für die angegebene Kamera und bereitet sie für die GUI auf."""
        data_list = []
        all_configs = ConfigManager.load_camera_config()
        global_rects = all_configs.get("Camera_ALL", {}).get("reference_rectangles", [])
        camera_pixels = all_configs.get(camera_name, {}).get("rectangle_pixels", {})

        main_room = next((r for r in global_rects if r.get("internal_id") == "MAIN_ROOM_CALIB"), None)
        if main_room:
            r_copy = dict(main_room)
            r_copy["is_main_calib"] = True
            pixel_info = camera_pixels.get("MAIN_ROOM_CALIB", [])
            for i, corner in enumerate(r_copy.get("corners", [])):
                if i < len(pixel_info):
                    corner["px"], corner["py"] = pixel_info[i].get("px"), pixel_info[i].get("py")
                else:
                    corner["px"], corner["py"] = None, None
            data_list.append(r_copy)

        for rect in global_rects:
            if rect.get("internal_id") == "MAIN_ROOM_CALIB": continue

            r_copy = dict(rect)
            if "internal_id" not in r_copy:
                r_copy["internal_id"] = str(r_copy.get("id", uuid.uuid4()))

            pixel_info = camera_pixels.get(r_copy["internal_id"], [])
            for i, corner in enumerate(r_copy.get("corners", [])):
                if i < len(pixel_info):
                    corner["px"], corner["py"] = pixel_info[i].get("px"), pixel_info[i].get("py")
                else:
                    corner["px"], corner["py"] = None, None
            data_list.append(r_copy)

        return data_list

    @staticmethod
    def extract_zones(rectangles_data: list) -> tuple:
        """Extrahiert und sortiert die Zonen
        (Dead und Mirror) aus der Liste aller Rechtecke."""
        dead_zones, mirror_zones = [], []
        for r in rectangles_data:
            if not r.get("is_zone"): continue
            pts = [[int(c["px"]), int(c["py"])] for c in r.get("corners", []) if c.get("px") is not None]
            if len(pts) == 4:
                if r.get("type") == "Dead Zone":
                    dead_zones.append(pts)
                else:
                    mirror_zones.append(pts)
        return dead_zones, mirror_zones

    @staticmethod
    def build_save_payloads(rectangles_data: list, deleted_rect_ids: list) -> tuple:
        """Bereitet die Payloads für das Speichern der Rechtecke vor,"""
        custom_rects_and_room = [r for r in rectangles_data if not r.get("is_zone")]

        global_rects_payload = []
        pixel_mapping_payload = {}

        for r in custom_rects_and_room:
            try:
                size_val = float(str(r.get("size_cm", "0")).replace(',', '.'))
            except ValueError:
                size_val = 0.0

            global_rects_payload.append({
                "internal_id": str(r.get("internal_id")),
                "display_id": str(r.get("display_id")),
                "type": str(r.get("type")),
                "size_cm": size_val,
                "is_active": bool(r.get("is_active", True)),
                "corners": [{"label": str(c["label"]), "x": float(c["x"]), "y": float(c["y"]), "z": float(c["z"])} for c
                            in r.get("corners", [])]
            })

            pixel_mapping_payload[str(r.get("internal_id"))] = [
                {"px": int(c["px"] or 0), "py": int(c["py"] or 0)} for c in r.get("corners", [])
            ]

        for del_id in deleted_rect_ids:
            global_rects_payload.append({"internal_id": del_id, "_delete": True})

        return global_rects_payload, pixel_mapping_payload