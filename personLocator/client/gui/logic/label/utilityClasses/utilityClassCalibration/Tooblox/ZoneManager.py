import logging
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex
from client.utils.ConfigManager import ConfigManager


class ZoneManager(QObject):
    """Verwaltet Dead- und Mirror-Zonen inkl. Thread-Sicherheit und Speicherung."""
    zones_changed = pyqtSignal()

    def __init__(self, client_name: str):
        super().__init__()
        self.client_name = client_name
        self.mutex = QMutex()
        self.dead_zones = []
        self.mirror_zones = []

    def load_zones(self, dead_zones: list, mirror_zones: list):
        """Initialisiert die Zonen aus dem ConfigLoader beim Start."""
        self.mutex.lock()
        try:
            self.dead_zones = dead_zones
            self.mirror_zones = mirror_zones
        finally:
            self.mutex.unlock()

    def get_zones(self) -> tuple:
        """Gibt eine sichere Kopie der Zonen für den Renderer zurück."""
        self.mutex.lock()
        try:
            return list(self.dead_zones), list(self.mirror_zones)
        finally:
            self.mutex.unlock()

    def add_dead_zone(self, points: list):
        if len(points) >= 3:
            self.mutex.lock()
            self.dead_zones.append(points)
            self.mutex.unlock()
            self._save_and_emit()

    def add_mirror_zone(self, points: list):
        if len(points) >= 3:
            self.mutex.lock()
            self.mirror_zones.append(points)
            self.mutex.unlock()
            self._save_and_emit()

    def delete_last_zone(self, zone_type: str = "dead"):
        changed = False
        self.mutex.lock()
        try:
            if zone_type == "dead" and self.dead_zones:
                self.dead_zones.pop()
                changed = True
            elif zone_type == "mirror" and self.mirror_zones:
                self.mirror_zones.pop()
                changed = True
        finally:
            self.mutex.unlock()

        if changed:
            self._save_and_emit()

    def clear_zones(self):
        self.mutex.lock()
        self.dead_zones = []
        self.mirror_zones = []
        self.mutex.unlock()
        self._save_and_emit()

    def _save_and_emit(self):
        """Speichert die Zonen via ConfigManager in die Datenbank und feuert das Update-Signal."""

        def clean_data(data):
            if isinstance(data, np.ndarray): return data.tolist()
            if isinstance(data, tuple): return list(data)
            if isinstance(data, list): return [clean_data(x) for x in data]
            return data

        try:
            self.mutex.lock()
            try:
                update_data = {
                    "dead_zones": clean_data(self.dead_zones),
                    "mirror_zones": clean_data(self.mirror_zones)
                }
            finally:
                self.mutex.unlock()

            ConfigManager.update_camera_settings(self.client_name, update_data)
            self.zones_changed.emit()
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Zonen: {e}")