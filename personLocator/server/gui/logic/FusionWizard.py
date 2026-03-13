from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QListWidgetItem, QLabel, QLineEdit, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt
import numpy as np


class FusionWizard(QDialog):
    """Dieser Code wird auch noch überarbeitet, da er aktuell nur eine manuelle Zuordnung ermöglicht,
    aber keine intelligente Vorschläge oder Kalibrierungs-Optionen bietet. Er dient aktuell als einfacher Fallback,
    falls die automatische Zuordnung versagt."""
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Personen Zuordnung (Keine Kalibrierung)")
        self.resize(500, 600)
        self.setStyleSheet("background-color: #222; color: #EEE;")

        self.setup_ui()
        self.load_tracks()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Info
        layout.addWidget(QLabel("1. Wähle alle IDs, die zu dieser Person gehören:"))

        # Liste mit Checkboxen
        self.list_tracks = QListWidget()
        self.list_tracks.setStyleSheet("background-color: #333; border: 1px solid #555;")
        layout.addWidget(self.list_tracks)

        # Name Input
        layout.addWidget(QLabel("2. Name der Person (für Datenbank):"))
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("z.B. Jonas")
        self.inp_name.setStyleSheet("padding: 5px; color: #FFF; background: #444;")
        layout.addWidget(self.inp_name)

        # Buttons
        btn_box = QHBoxLayout()
        btn_merge = QPushButton("Person Zuweisen & Speichern")
        btn_merge.setStyleSheet("background-color: #008800; padding: 10px; font-weight: bold;")
        btn_merge.clicked.connect(self.on_merge)

        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_merge)
        layout.addLayout(btn_box)

    def load_tracks(self):
        """Holt aktuelle Rohdaten und filtert Nullen heraus."""
        if not self.controller:
            return

        raw_data = self.controller.get_unmerged_tracks()
        self.list_tracks.clear()

        for cam, persons in raw_data.items():
            for p in persons:
                pid = p.get('id', '?')

                pos = p.get('pos', np.array([0, 0, 0]))
                if not isinstance(pos, (list, tuple, np.ndarray)):
                    pos = np.array([0, 0, 0])

                if isinstance(pos, np.ndarray):
                    pos_list = pos.tolist()
                else:
                    pos_list = pos

                is_valid = not (abs(pos_list[0]) < 1 and abs(pos_list[1]) < 1 and abs(pos_list[2]) < 1)
                pos_str = f"[{int(pos_list[0])}, {int(pos_list[1])}, {int(pos_list[2])}]"

                if is_valid:
                    text = f"{cam} - ID: {pid} @ {pos_str}"
                    item = QListWidgetItem(text)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    text = f"{cam} - ID: {pid} @ {pos_str} (KEIN SIGNAL)"
                    item = QListWidgetItem(text)
                    item.setForeground(Qt.GlobalColor.red)
                    item.setFlags(Qt.ItemFlag.NoItemFlags)

                item_data = {'cam': cam, 'id': pid, 'pos': pos_list, 'height': p.get('height', 0)}
                item.setData(Qt.ItemDataRole.UserRole, item_data)

                self.list_tracks.addItem(item)

    def on_merge(self):
        """Sammelt die ausgewählten IDs und den Namen, validiert die Eingaben und sendet sie an den Controller zur manuellen Zuordnung."""
        name = self.inp_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Fehler", "Bitte einen Namen eingeben!")
            return

        selection = []
        for i in range(self.list_tracks.count()):
            item = self.list_tracks.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selection.append(item.data(Qt.ItemDataRole.UserRole))

        if len(selection) < 1:
            QMessageBox.warning(self, "Fehler", "Bitte mindestens eine ID auswählen.")
            return

        self.controller.execute_manual_merge(name, selection)
        self.accept()