import time
import numpy as np


class GlobalPerson:
    """
    Repräsentiert eine 'echte' Person im Raum.
    Kann Daten aus der DB laden und speichern.
    Hält Live-Daten aller einzelnen Clients.
    """

    def __init__(self, temp_id: int, pos: np.ndarray, height: float):
        self.id = temp_id
        self.name = "Unknown"
        self.db_ref = None

        # --- Fusionierte Live-Daten ---
        self.pos = np.array(pos)
        self.height = height
        self.width = 45.0

        # --- Rohdaten der Clients  ---
        self.client_observations = {}
        self.last_update = time.time()

        # --- Detaillierte Eigenschaften (zum Speichern) ---
        self.fixed_height = None

        # Farb-Profil: Speichert Farben pro Keypoint für Vorne/Hinten
        self.color_profile = {"front": {}, "back": {}}

        # Alle Keypoints (fusioniert oder vom besten Client)
        self.keypoints = {}

    def set_database(self, db_instance):
        """Verbindet diese Person mit der Datenbank."""
        self.db_ref = db_instance

    def load_identity(self, name):
        """
        Lädt gespeicherte Daten (Größe, Breite, Farben) aus der DB.
        """
        if not self.db_ref: return

        data = self.db_ref.get_person_data(name)
        if data:
            self.name = name
            if "fixed_height" in data:
                self.fixed_height = data["fixed_height"]
                self.height = self.fixed_height
            elif "height" in data:
                self.height = data["height"]

            if "width" in data:
                self.width = data["width"]

            if "color_profile" in data:
                self.color_profile = data["color_profile"]

            print(f"GlobalPerson {self.id}: Identität '{name}' geladen.")

    def save_identity(self, name=None, fix_height=None):
        """
        Speichert den aktuellen Status in die Datenbank.
        """
        if not self.db_ref: return

        target_name = name if name else self.name
        if target_name == "Unknown": return

        if fix_height is not None:
            self.fixed_height = fix_height
            self.height = fix_height

        save_data = {
            "height": float(self.height),
            "width": float(self.width),
            "fixed_height": self.fixed_height,
            "color_profile": self.color_profile,
            "last_seen_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.db_ref.update_person(target_name, save_data)
        self.name = target_name


    def update(self, new_pos, new_height, cam_name, client_id, raw_pos, keypoints, orientation="front", raw_data=None):
        """
        Haupteingang für neue Daten von einer Kamera.
        """
        self.pos = self.pos * 0.6 + np.array(new_pos) * 0.4

        if self.fixed_height is not None:
            self.height = self.fixed_height
        else:
            self.height = self.height * 0.9 + new_height * 0.1

        self.last_update = time.time()

        if cam_name not in self.client_observations:
            self.client_observations[cam_name] = {
                "id": client_id,
                "pos": raw_pos,
                "last_seen": time.time(),
                "raw_data": raw_data or {},
                "last_kps": {},
                "last_skel_3d": {}
            }

        obs = self.client_observations[cam_name]
        obs["id"] = client_id
        obs["pos"] = raw_pos
        obs["last_seen"] = time.time()
        obs["raw_data"] = raw_data or {}

        for k_id in obs["last_kps"]:
            obs["last_kps"][k_id]["c"] = 0.0

        if raw_data:
            for kp in raw_data.get("keypoints", []):
                obs["last_kps"][int(kp["id"])] = kp.copy()

            for j_id, pos3d in raw_data.get("skeleton_3d", {}).items():
                obs["last_skel_3d"][int(j_id)] = pos3d

        self.keypoints = keypoints
        self._learn_colors(keypoints, orientation)

    def _learn_colors(self, keypoints, orientation):
        """
        Sammelt Farbdaten pro Keypoint.
        orientation: "front", "back", "side"
        """
        if orientation not in ["front", "back"]: return

        target_profile = self.color_profile[orientation]

        for kp in keypoints:
            k_id = str(kp['id'])

            if 'color' in kp and kp['c'] > 0.6:
                new_col = np.array(kp['color'])

                if k_id in target_profile:
                    old_col = np.array(target_profile[k_id])
                    mixed_col = old_col * 0.9 + new_col * 0.1
                    target_profile[k_id] = mixed_col.tolist()
                else:
                    target_profile[k_id] = new_col.tolist()