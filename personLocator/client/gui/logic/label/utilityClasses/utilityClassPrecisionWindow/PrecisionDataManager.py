import uuid
from client.utils.ConfigManager import ConfigManager


class PrecisionDataManager:
    @staticmethod
    def load_all_rectangles(camera_name: str) -> list:
        """Lädt alle Rechtecke und berücksichtigt ab sofort den Kameraspezifischen is_active Status!"""
        data_list = []
        all_configs = ConfigManager.load_camera_config()
        global_rects = all_configs.get("Camera_ALL", {}).get("reference_rectangles", [])
        camera_pixels = all_configs.get(camera_name, {}).get("rectangle_pixels", {})

        main_room = next((r for r in global_rects if r.get("internal_id") == "MAIN_ROOM_CALIB"), None)
        if main_room:
            r_copy = dict(main_room)
            r_copy["is_main_calib"] = True

            cam_data = camera_pixels.get("MAIN_ROOM_CALIB", {})
            if isinstance(cam_data, dict) and "pixels" in cam_data:
                r_copy["is_active"] = cam_data.get("is_active", True)
                pixel_info = cam_data.get("pixels", [])
            else:
                pixel_info = cam_data if isinstance(cam_data, list) else []

            for i, corner in enumerate(r_copy.get("corners", [])):
                if i < len(pixel_info):
                    corner["px"] = pixel_info[i].get("px")
                    corner["py"] = pixel_info[i].get("py")
            data_list.append(r_copy)

        for rect in global_rects:
            if rect.get("internal_id") == "MAIN_ROOM_CALIB" or rect.get("type") == "Haupt-Kalibrierung":
                continue

            r_copy = dict(rect)
            r_id = r_copy["internal_id"]

            cam_data = camera_pixels.get(r_id, {})
            if isinstance(cam_data, dict) and "pixels" in cam_data:
                r_copy["is_active"] = cam_data.get("is_active", False)
                pixel_info = cam_data.get("pixels", [])
            else:
                r_copy["is_active"] = False
                pixel_info = cam_data if isinstance(cam_data, list) else []

            for i, corner in enumerate(r_copy.get("corners", [])):
                if i < len(pixel_info):
                    corner["px"] = pixel_info[i].get("px")
                    corner["py"] = pixel_info[i].get("py")
            data_list.append(r_copy)

        return data_list

    @staticmethod
    def extract_zones(rectangles_data: list) -> tuple:
        """Filtert die Live-Zonen heraus und trennt sie in Dead- und Mirror-Zonen."""
        dead_zones = []
        mirror_zones = []

        for r in rectangles_data:
            if r.get("is_zone"):
                if r.get("type") == "Dead Zone":
                    dead_zones.append(r)
                elif r.get("type") == "Mirror Zone":
                    mirror_zones.append(r)

        return dead_zones, mirror_zones

    @staticmethod
    def build_save_payloads(rectangles_data: list, deleted_ids: set) -> tuple:
        """Baut getrennte Payloads: 1. Globale Raummaße 2. Lokale Pixel und bindet ihn an die Kamera!"""
        custom_rects_and_room = [r for r in rectangles_data if not r.get("is_zone")]

        global_rects_payload = []
        pixel_mapping_payload = {}

        def safe_float(val):
            if val is None: return 0.0
            try:
                return float(val)
            except ValueError:
                return 0.0

        for r in custom_rects_and_room:
            try:
                size_val = float(str(r.get("size_cm", "0")).replace(',', '.'))
            except ValueError:
                size_val = 0.0

            actual_active_status = bool(r.get("is_active", True))

            global_corners = []
            for c in r.get("corners", []):
                global_corners.append({
                    "label": str(c.get("label", "")),
                    "x": safe_float(c.get("x")),
                    "y": safe_float(c.get("y")),
                    "z": safe_float(c.get("z"))
                })

            global_rects_payload.append({
                "internal_id": str(r.get("internal_id")),
                "display_id": str(r.get("display_id")),
                "type": str(r.get("type")),
                "size_cm": size_val,
                "is_active": True,
                "corners": global_corners
            })

            pixels_to_send = []
            if actual_active_status:
                for c in r.get("corners", []):
                    px_val = c.get("px")
                    py_val = c.get("py")

                    # 'None' wird zu 0 normalisiert, damit der Server immer valide Pixelwerte erhält.
                    safe_px = int(px_val) if px_val is not None else 0
                    safe_py = int(py_val) if py_val is not None else 0

                    pixels_to_send.append({
                        "px": safe_px,
                        "py": safe_py,
                        "label": str(c.get("label", ""))
                    })

            pixel_mapping_payload[str(r.get("internal_id"))] = {
                "is_active": actual_active_status,
                "pixels": pixels_to_send
            }

        for d_id in deleted_ids:
            global_rects_payload.append({"internal_id": d_id, "_delete": True})

        return global_rects_payload, pixel_mapping_payload