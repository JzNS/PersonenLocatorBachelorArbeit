import uuid
from PyQt6.QtCore import QObject, pyqtSignal, Qt

class PrecisionInteractionHandler(QObject):
    """
    Verwaltet die Zustände der Maus-Interaktion (Zeichnen von Zonen vs. Setzen von Pixeln).
    """
    # Signale, um das Hauptfenster zu informieren, was es tun soll
    zone_info_changed = pyqtSignal(str)
    cursor_changed = pyqtSignal(Qt.CursorShape)
    redraw_requested = pyqtSignal()
    zone_completed = pyqtSignal(dict)
    pixel_click_requested = pyqtSignal(object, object, bool) # x, y, is_delete

    def __init__(self):
        super().__init__()
        self.draw_mode = None
        self.temp_points = []

    def start_zone_drawing(self, mode: str):
        """Aktiviert den Zonen-Zeichenmodus."""
        self.draw_mode = mode
        self.temp_points = []
        self.cursor_changed.emit(Qt.CursorShape.CrossCursor)
        self.zone_info_changed.emit(f"{mode.upper()}: Klicke 4 Eckpunkte")

    def handle_left_click(self, x: int, y: int):
        """Verarbeitet einen Linksklick (Punkt setzen)."""
        if self.draw_mode:
            self.temp_points.append((x, y))
            pts_count = len(self.temp_points)
            self.zone_info_changed.emit(f"{self.draw_mode.upper()}: Punkt {pts_count}/4")

            if pts_count == 4:
                new_zone = self._create_zone_dict(self.draw_mode, self.temp_points)
                self.zone_completed.emit(new_zone)
                self.zone_info_changed.emit(f"{self.draw_mode} gespeichert!")

                self.draw_mode = None
                self.temp_points = []
                self.cursor_changed.emit(Qt.CursorShape.ArrowCursor)

            self.redraw_requested.emit()
        else:
            self.pixel_click_requested.emit(x, y, False)

    def handle_right_click(self):
        """Verarbeitet einen Rechtsklick (Abbrechen oder Löschen)."""
        if self.draw_mode:
            self.draw_mode = None
            self.temp_points = []
            self.cursor_changed.emit(Qt.CursorShape.ArrowCursor)
            self.zone_info_changed.emit("Zeichnen abgebrochen.")
            self.redraw_requested.emit()
        else:
            self.pixel_click_requested.emit(None, None, True)

    def _create_zone_dict(self, z_type: str, points: list):
        """Erzeugt das standardisierte Dictionary für eine neue Zone."""
        return {
            "internal_id": str(uuid.uuid4()),
            "display_id": f"Z-{uuid.uuid4().hex[:4]}",
            "type": z_type,
            "size_cm": 0,
            "is_zone": True,
            "corners": [
                {"label": f"P{i + 1}", "x": 0.0, "y": 0.0, "z": 0.0, "px": p[0], "py": p[1]}
                for i, p in enumerate(points)
            ]
        }