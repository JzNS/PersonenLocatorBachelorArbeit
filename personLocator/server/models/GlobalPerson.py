import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union


@dataclass(slots=True)
class ClientObservation:
    """Repräsentiert die Rohdaten eines einzelnen Clients für eine Person."""
    id: int
    pos: np.ndarray
    last_seen: float
    raw_data: dict[str, Any] = field(default_factory=dict)
    last_kps: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_skel_3d: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class GlobalPerson:
    """
    Repräsentiert eine 'echte' Person im Raum.
    Kann Daten aus der DB laden und speichern.
    Hält Live-Daten aller einzelnen Clients.
    """
    id: int
    pos: np.ndarray
    height: float
    name: str = "Unknown"
    db_ref: Any = field(default=None, repr=False)

    width: float = 45.0
    fusion_skel: dict[int, np.ndarray] = field(default_factory=dict)
    skeleton_3d: dict[int, np.ndarray] = field(default_factory=dict)
    forward_vec: Optional[np.ndarray] = None
    fused_height: float = 175.0
    fused_width: float = 45.0

    client_observations: dict[str, ClientObservation] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)

    fixed_height: Optional[float] = None

    # Farb-Profil: Speichert Farben pro Keypoint für Vorne/Hinten
    color_profile: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {"front": {}, "back": {}}
    )

    keypoints: list[dict[str, Any]] = field(default_factory=list)

    current_pointers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Liste von (timestamp, position_np_ndarray, label)
    pointing_history: list[dict[str, Any]] = field(default_factory=list)

    def set_database(self, db_instance: Any) -> None:
        """Verbindet diese Person mit der Datenbank."""
        self.db_ref = db_instance

    def load_identity(self, name: str) -> None:
        """
        Lädt gespeicherte Daten (Größe, Breite, Farben) aus der DB.
        """
        if not self.db_ref:
            return

        data: Optional[dict[str, Any]] = self.db_ref.get_person_data(name)
        if data:
            self.name = name
            if "fixed_height" in data:
                self.fixed_height = data["fixed_height"]
                if self.fixed_height is not None:
                    self.height = self.fixed_height
            elif "height" in data:
                self.height = data["height"]

            if "width" in data:
                self.width = data["width"]

            if "color_profile" in data:
                self.color_profile = data["color_profile"]

            print(f"GlobalPerson {self.id}: Identität '{name}' geladen.")

    def save_identity(self, name: Optional[str] = None, fix_height: Optional[float] = None) -> None:
        """
        Speichert den aktuellen Status in die Datenbank.
        """
        if not self.db_ref:
            return

        target_name: str = name if name else self.name
        if target_name == "Unknown":
            return

        if fix_height is not None:
            self.fixed_height = fix_height
            self.height = fix_height

        save_data: dict[str, Any] = {
            "height": float(self.height),
            "width": float(self.width),
            "fixed_height": self.fixed_height,
            "color_profile": self.color_profile,
            "last_seen_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.db_ref.update_person(target_name, save_data)
        self.name = target_name

    def update(self, new_pos: Union[np.ndarray, list[float]], new_height: float, cam_name: str,
               client_id: int, raw_pos: np.ndarray, keypoints: list[dict[str, Any]],
               orientation: str = "front", raw_data: Optional[dict[str, Any]] = None) -> None:
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
            self.client_observations[cam_name] = ClientObservation(
                id=client_id,
                pos=raw_pos,
                last_seen=time.time(),
                raw_data=raw_data or {},
                last_kps={},
                last_skel_3d={}
            )

        obs: ClientObservation = self.client_observations[cam_name]
        obs.id = client_id
        obs.pos = raw_pos
        obs.last_seen = time.time()
        obs.raw_data = raw_data or {}

        for k_id in obs.last_kps:
            obs.last_kps[k_id]["c"] = 0.0

        if raw_data:
            for kp in raw_data.get("keypoints", []):
                obs.last_kps[int(kp["id"])] = kp.copy()

            for j_id, pos3d in raw_data.get("skeleton_3d", {}).items():
                obs.last_skel_3d[int(j_id)] = pos3d

        self.keypoints = keypoints
        self._learn_colors(keypoints, orientation)

    def _learn_colors(self, keypoints: list[dict[str, Any]], orientation: str) -> None:
        """
        Sammelt Farbdaten pro Keypoint.
        orientation: "front", "back", "side"
        """
        if orientation not in ["front", "back"]:
            return

        target_profile: dict[str, list[float]] = self.color_profile[orientation]

        for kp in keypoints:
            k_id: str = str(kp['id'])

            if 'color' in kp and kp['c'] > 0.6:
                new_col: np.ndarray = np.array(kp['color'])

                if k_id in target_profile:
                    old_col: np.ndarray = np.array(target_profile[k_id])
                    mixed_col: np.ndarray = old_col * 0.9 + new_col * 0.1
                    target_profile[k_id] = mixed_col.tolist()
                else:
                    target_profile[k_id] = new_col.tolist()