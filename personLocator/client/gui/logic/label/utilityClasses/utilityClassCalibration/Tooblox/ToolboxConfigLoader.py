import logging
import uuid

from client.gui.logic.label.math.GeometryMath import GeometryMath
from client.utils.ConfigManager import ConfigManager


class ToolboxConfigLoader:
    """
    Kapselt die gesamte Logik zum Auslesen, Formatieren und Zusammenbauen
    der Kalibrierungsdaten aus dem ConfigManager.
    """

    @staticmethod
    def load_full_state(client_name: str) -> dict:
        """Lädt alle Daten und gibt ein fertiges Zustands-Dictionary zurück."""
        state = {
            "view_options": {},
            "real_room_dims": {"width": 600.0, "height": 250.0, "depth": 800.0},
            "dead_zones": [],
            "mirror_zones": [],
            "custom_rectangles": [],
            "points_3d_data": []
        }

        try:
            # 1. Basis-Settings holen
            settings = ConfigManager.get_camera_settings(client_name)

            # 2. View Options verpacken
            state["view_options"] = {
                "show_real_world": settings.get("view_show_real", True),
                "show_camera_world": settings.get("view_show_grid", True),
                "show_skeleton_3d": settings.get("view_show_skeleton", True),
                "render_3d_enabled": settings.get("view_render_enabled", True),
                "show_floor_grid": settings.get("view_show_floor_grid", True),
                "show_rays": settings.get("view_show_rays", True),
                "show_sightlines": settings.get("view_show_sightlines", True),
                "performance_mode": settings.get("performance_mode", False)
            }

            # 3. Linsen-Profil für PnP (Globale Mathematik injizieren)
            matrix = settings.get("camera_matrix")
            dist = settings.get("dist_coeffs")
            profile_name = settings.get("active_lens_profile")

            if not matrix and profile_name:
                all_profiles = ConfigManager.get_camera_settings("Camera_ALL")
                profile = all_profiles.get(profile_name, {})
                matrix = profile.get("camera_matrix")
                dist = profile.get("dist_coeffs")

            if matrix is not None and dist is not None:
                GeometryMath.camera_matrix = matrix
                GeometryMath.dist_coeffs = dist
                logging.info(f"PnP-Mathe: Profil '{profile_name or 'Custom'}' geladen.")
            else:
                GeometryMath.camera_matrix = None
                GeometryMath.dist_coeffs = None

            # 4. Raummaße
            all_configs = ConfigManager.load_camera_config()
            global_data = all_configs.get("Camera_ALL", {})
            dims = global_data.get("room_dimensions", {})
            state["real_room_dims"] = {
                "width": float(dims.get("width", 600.0)),
                "height": float(dims.get("height", 250.0)),
                "depth": float(dims.get("depth", 800.0))
            }

            # 5. Zonen
            raw_dead = settings.get("dead_zones", [])
            raw_mirror = settings.get("mirror_zones", [])
            state["dead_zones"] = [z for z in raw_dead if isinstance(z, list) and len(z) >= 3]
            state["mirror_zones"] = [z for z in raw_mirror if isinstance(z, list) and len(z) >= 3]

            # 6. Raum-Punkte (Hauptkalibrierung)
            points = ConfigManager.load_camera_points(client_name)
            if points:
                state["points_3d_data"] = points

            # 7. Custom Vierecke zusammenbauen (Inklusive UUID-Check und Filter)
            global_rects = ConfigManager.load_global_rectangles()
            camera_pixels = ConfigManager.load_camera_rectangle_pixels(client_name)

            custom_rects = []
            for rect in global_rects:
                if rect.get("internal_id") == "MAIN_ROOM_CALIB":
                    continue

                rect_copy = dict(rect)
                internal_id = rect_copy.get("internal_id")
                if not internal_id:
                    internal_id = str(uuid.uuid4())
                    rect_copy["internal_id"] = internal_id

                pixel_info = camera_pixels.get(internal_id, [])

                for i, corner in enumerate(rect_copy.get("corners", [])):
                    if i < len(pixel_info):
                        corner["px"] = pixel_info[i].get("px")
                        corner["py"] = pixel_info[i].get("py")
                    else:
                        corner["px"], corner["py"] = None, None

                custom_rects.append(rect_copy)

            state["custom_rectangles"] = custom_rects

        except Exception as e:
            logging.error(f"ToolboxConfigLoader Fehler: {e}")

        return state